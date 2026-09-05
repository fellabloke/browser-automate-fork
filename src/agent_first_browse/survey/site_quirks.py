"""URL-scoped, deterministic workarounds for known survey-provider defects."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


logger = logging.getLogger(__name__)
_REGISTRY_PATH = Path(__file__).with_name("survey_site_quirks.json")
_UK_FULL_POSTCODE = re.compile(
    r"^([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})$",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def load_quirk_registry() -> tuple[dict[str, Any], ...]:
    """Load the reviewed data-only registry once, failing closed if malformed."""
    try:
        document = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        quirks = document.get("quirks", [])
        if not isinstance(quirks, list):
            raise ValueError("quirks must be a list")
        return tuple(rule for rule in quirks if isinstance(rule, dict))
    except Exception as exc:  # noqa: BLE001 - a bad optional rule must not stop a run
        logger.warning("Survey quirk registry unavailable: %s", exc)
        return ()


def _hostname(url: str) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _host_matches(hostname: str, suffix: str) -> bool:
    suffix = str(suffix or "").lower().strip().lstrip(".").rstrip(".")
    return bool(suffix and (hostname == suffix or hostname.endswith("." + suffix)))


def matching_site_quirks(url: str) -> tuple[dict[str, Any], ...]:
    """Return rules for an exact hostname or a real subdomain boundary."""
    hostname = _hostname(url)
    if not hostname:
        return ()
    return tuple(
        rule for rule in load_quirk_registry()
        if any(_host_matches(hostname, suffix)
               for suffix in rule.get("host_suffixes", []))
    )


def uk_postcode_outward(value: str) -> str:
    """Return the outward code for a valid full UK postcode; otherwise unchanged."""
    original = str(value or "")
    compact = re.sub(r"\s+", "", original).upper()
    match = _UK_FULL_POSTCODE.fullmatch(compact)
    return match.group(1).upper() if match else original


def _normalise_signal(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _target_is_postcode_field(
    action: dict[str, Any],
    selector_map: dict[str, dict],
    page_text: str,
    field_terms: list[str],
) -> bool:
    element_id = str(action.get("element_id") or "")
    element = selector_map.get(element_id) or {}
    direct = " ".join(str(value or "") for value in (
        action.get("target_name"),
        element.get("text"),
        element.get("name"),
        element.get("aria_label"),
        element.get("hint"),
        element.get("description"),
    ))
    normalised_terms = tuple(_normalise_signal(term) for term in field_terms)
    direct_signal = _normalise_signal(direct)
    if any(term and term in direct_signal for term in normalised_terms):
        return True

    # Some custom inputs expose no useful label. Fall back only when the page
    # visibly asks for a postcode and the proposed target is its sole text input.
    generic_signal = direct_signal
    for generic in (
        "input", "empty", "textbox", "textfield", "field", "entervalue",
        "typehere", "youranswer", "pleaseenter",
    ):
        generic_signal = generic_signal.replace(generic, "")
    if generic_signal:
        return False
    if not any(term and term in _normalise_signal(page_text) for term in normalised_terms):
        return False
    input_ids = [
        str(eid) for eid, item in selector_map.items()
        if item.get("kind") == "input"
        and str(item.get("control_type") or "").lower()
        not in {"checkbox", "radio", "submit", "button"}
    ]
    return len(input_ids) == 1 and element_id == input_ids[0]


def apply_site_quirks_to_action(
    action: dict[str, Any],
    *,
    url: str,
    selector_map: dict[str, dict] | None = None,
    page_text: str = "",
) -> tuple[dict[str, Any], str]:
    """Apply one matching, field-grounded transform without mutating the proposal."""
    transformed = dict(action or {})
    selector_map = selector_map or {}
    for rule in matching_site_quirks(url):
        if transformed.get("verb") != rule.get("action"):
            continue
        if not _target_is_postcode_field(
            transformed,
            selector_map,
            page_text,
            list(rule.get("field_terms", [])),
        ):
            continue
        if rule.get("transform") != "uk_postcode_outward":
            continue
        before = str(transformed.get("text") or "")
        after = uk_postcode_outward(before)
        if after == before:
            continue
        transformed["text"] = after
        transformed["site_quirk_applied"] = rule.get("id", "unknown")
        target = selector_map.get(str(transformed.get("element_id") or "")) or {}
        current = str(target.get("value") or "").strip()
        if not current:
            filled = re.search(
                r"filled:\s*[\"']?([^\"'\]]+)",
                " ".join(str(target.get(field) or "") for field in ("text", "name", "hint")),
                re.IGNORECASE,
            )
            current = filled.group(1).strip() if filled else ""
        if current and _normalise_signal(current) != _normalise_signal(after):
            transformed["force_retype"] = True
            transformed["replace_existing"] = True
        return transformed, str(rule.get("id") or "unknown")
    return transformed, ""


def render_site_quirk_guidance(url: str) -> str:
    """Render authoritative guidance for every fallback model on this provider."""
    matched = matching_site_quirks(url)
    if not matched:
        return ""
    lines = ["═══ URL-SCOPED SURVEY PROVIDER QUIRKS — AUTHORITATIVE ═══"]
    for rule in matched:
        guidance = str(rule.get("guidance") or "").strip()
        if guidance:
            lines.append(f"• [{rule.get('id', 'unknown')}] {guidance}")
    lines.append("These rules are deterministic provider workarounds and outrank generic model assumptions.")
    return "\n".join(lines)


def fresh_dashboard_after_completion(url: str) -> str:
    """Return a fresh-tab dashboard for every verified provider completion."""
    for rule in matching_site_quirks(url):
        if rule.get("event") != "survey_completed":
            continue
        destination = str(rule.get("fresh_dashboard_url") or "").strip()
        if destination:
            return destination
    # The caller supplies the already-established provider home, never the
    # external survey URL. Reopening that exact HTTP(S) route is a safe generic
    # reset for panels without a data-file rule and prevents stale completion
    # overlays from consuming hundreds of actions.
    candidate = str(url or "").strip()
    return candidate if candidate.startswith(("https://", "http://")) else ""


def fresh_dashboard_after_boundary(url: str, outcome: str = "") -> str:
    """Return a provider dashboard that must be recreated at any boundary."""
    for rule in matching_site_quirks(url):
        if not rule.get("fresh_dashboard_after_any_boundary"):
            continue
        destination = str(rule.get("fresh_dashboard_url") or "").strip()
        if destination:
            return destination
    if str(outcome or "").startswith("completed"):
        return fresh_dashboard_after_completion(url)
    return ""


def provider_start_action(
    url: str,
    page_text: str,
    selector_map: dict[str, dict] | None = None,
) -> tuple[str, str]:
    """Resolve reviewed multi-stage panel entry controls without an LLM call."""
    hostname = _hostname(url)
    if not _host_matches(hostname, "primeopinion.com"):
        return "", ""
    selector_map = selector_map or {}
    priorities = (
        ("participate", 0),
        ("start the survey", 1),
        ("start survey", 2),
        ("start your first survey", 3),
    )
    ranked: list[tuple[int, str, str]] = []
    for element_id, element in selector_map.items():
        if element.get("disabled") or element.get("visible") is False:
            continue
        kind = str(element.get("kind") or "").lower()
        if kind not in {"button", "link", "other"}:
            continue
        label = " ".join(str(element.get(key) or "") for key in (
            "text", "name", "aria_label", "hint",
        )).strip().lower()
        for phrase, score in priorities:
            if phrase in label:
                ranked.append((score, str(element_id), phrase))
                break
    if not ranked:
        return "", ""
    _score, element_id, stage = min(ranked)
    return element_id, stage


def qmee_active_survey_action(
    url: str,
    page_text: str,
    selector_map: dict[str, dict] | None = None,
    *,
    previous_boundary: str = "",
) -> tuple[str, str]:
    """Resolve Qmee's server-side active-survey marker deterministically.

    Closing local tabs does not clear Qmee's remote marker.  Its feedback page
    first asks whether to use the newly selected survey and may then ask what
    happened to the old one.  Return the exact live control for the applicable
    stage so this workflow never depends on a model interpreting emoji labels.
    """
    if not _host_matches(_hostname(url), "qmee.com"):
        return "", ""
    selector_map = selector_map or {}
    combined = " ".join((str(page_text or ""), *(
        " ".join(str(element.get(key) or "") for key in ("text", "name", "hint"))
        for element in selector_map.values()
    ))).lower()
    if "already doing a survey" not in combined:
        return "", ""

    labelled = []
    for element_id, element in selector_map.items():
        if element.get("disabled") is True:
            continue
        label = " ".join(str(element.get(key) or "") for key in (
            "text", "name", "aria_label", "hint",
        )).strip().lower()
        if label:
            labelled.append((str(element_id), label))

    reason_controls = [
        (element_id, label) for element_id, label in labelled
        if any(marker in label for marker in (
            "finished it already", "broken", "stuck", "too long",
            "didn't qualify", "did not qualify",
        ))
    ]
    if reason_controls:
        boundary = str(previous_boundary or "").lower()
        if boundary.startswith("completed"):
            preferred = ("finished it already",)
            reason = "completed"
        elif "too_long" in boundary or "too long" in boundary:
            preferred = ("too long",)
            reason = "too_long"
        else:
            preferred = ("broken", "stuck")
            reason = "broken_or_stuck"
        for marker in preferred:
            match = next((item for item in reason_controls if marker in item[1]), None)
            if match:
                return match[0], reason
        # Safe autonomous fallback: an uncleared session after a non-completion
        # boundary is a stale/broken survey, never a fabricated completion.
        match = next((item for item in reason_controls
                      if "broken" in item[1] or "stuck" in item[1]), None)
        if match:
            return match[0], "broken_or_stuck"

    proceed = next((
        (element_id, label) for element_id, label in labelled
        if "do this survey instead" in label or "continue with this survey" in label
    ), None)
    if proceed:
        return proceed[0], "use_new_survey"
    return "", ""
