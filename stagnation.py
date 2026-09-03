"""Stagnation Detector — progress-aware loop breaking (V29 / Phase 3, Mandate 3).

THE LOOP CURE
═════════════
The agent gets stuck "busy but not progressing": it keeps clicking/scrolling and
the DOM keeps changing (so the per-step critic sees "progress"), yet it makes no
real headway toward the goal — same URL forever, a flat goal-score, or a short
cycle of repeating actions (A,B,A,B…). The existing guards catch EXACT repeats
([overwatch] loop_signatures) and identical consecutive actions; they miss this
*productive-looking churn*.

Following PABU (Progress-Aware Belief Update, arXiv 2602.09138), we model progress
as a task-dependent but environment-agnostic abstraction and combine three cheap,
already-computed signals:

  1. URL-stuck     — `same_url_streak` (computed every step, previously UNUSED in
                     the live graph — revived here).
  2. Goal-flat     — the PRM goal-score window is flat (reuses cognition.detect_stall).
  3. Action-cycle  — the recent action signatures form a short repeating cycle.

When ≥2 fire, we declare stagnation and emit a directive that tells the agent it is
NOT making real progress toward its goal and must change approach (different
element, scroll, or re-navigate) — and, at the higher level, opens the eyes to
re-orient. Universal: no site/task rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

URL_STUCK_THRESHOLD = 4      # steps on the same URL before it counts as a signal
CYCLE_MIN_REPEATS = 2        # a 2-step cycle repeated ≥2× (A,B,A,B) = churn


@dataclass
class StagnationSignal:
    stuck: bool = False
    level: int = 0                       # number of signals that fired (0..3)
    reasons: list[str] = field(default_factory=list)
    note: str = ""                       # directive for the guidance bus


@dataclass
class NavigationCycleSignal:
    """A learned A→B→A→B navigation loop and the action that closes it."""

    detected: bool = False
    note: str = ""
    blocked_action: dict[str, Any] = field(default_factory=dict)


def _same_page_state(
    left_url: str, left_fingerprint: str,
    right_url: str, right_fingerprint: str,
) -> bool:
    """Compare URLs first and fingerprints when both sides provide one."""
    left_url = str(left_url or "").rstrip("/")
    right_url = str(right_url or "").rstrip("/")
    if not left_url or left_url != right_url:
        return False
    left_fingerprint = str(left_fingerprint or "")
    right_fingerprint = str(right_fingerprint or "")
    if left_fingerprint and right_fingerprint:
        return left_fingerprint == right_fingerprint
    return True


def detect_navigation_cycle(
    history: list[dict[str, Any]],
    *,
    current_url: str,
    current_fingerprint: str = "",
) -> NavigationCycleSignal:
    """Learn the loop-closing action from three alternating transitions.

    For A→B, B→A, A→B, the action used for B→A is unsafe to repeat on the
    current B page. Element identity is retained so other controls with the same
    ambiguous label remain available (PaidWork has several separate "Fill out"
    controls).
    """
    transitions = [
        item for item in list(history or [])
        if item.get("pre_url") and item.get("url")
        and not _same_page_state(
            item.get("pre_url", ""), item.get("pre_page_fingerprint", ""),
            item.get("url", ""), item.get("post_page_fingerprint", ""),
        )
    ]
    if len(transitions) < 3:
        return NavigationCycleSignal()
    first, reverse, repeated = transitions[-3:]
    cross_url_cycle = (
        str(first.get("pre_url") or "").rstrip("/")
        != str(first.get("url") or "").rstrip("/")
    )

    def matches(left_url: str, left_fp: str, right_url: str, right_fp: str) -> bool:
        if cross_url_cycle:
            return bool(
                str(left_url or "").rstrip("/")
                and str(left_url or "").rstrip("/") == str(right_url or "").rstrip("/")
            )
        return _same_page_state(left_url, left_fp, right_url, right_fp)

    a_matches = (
        matches(
            first.get("pre_url", ""), first.get("pre_page_fingerprint", ""),
            reverse.get("url", ""), reverse.get("post_page_fingerprint", ""),
        )
        and matches(
            reverse.get("url", ""), reverse.get("post_page_fingerprint", ""),
            repeated.get("pre_url", ""), repeated.get("pre_page_fingerprint", ""),
        )
    )
    b_matches = (
        matches(
            first.get("url", ""), first.get("post_page_fingerprint", ""),
            reverse.get("pre_url", ""), reverse.get("pre_page_fingerprint", ""),
        )
        and matches(
            reverse.get("pre_url", ""), reverse.get("pre_page_fingerprint", ""),
            repeated.get("url", ""), repeated.get("post_page_fingerprint", ""),
        )
        and matches(
            repeated.get("url", ""), repeated.get("post_page_fingerprint", ""),
            current_url, current_fingerprint,
        )
    )
    if not (a_matches and b_matches):
        return NavigationCycleSignal()

    blocked = {
        "page_url": str(current_url or ""),
        "verb": str(reverse.get("verb") or reverse.get("action") or "click").split("(", 1)[0],
        "element_id": reverse.get("element_id"),
        "target_name": str(reverse.get("target_name") or "")[:100],
        "target_context": str(reverse.get("target_context") or "")[:120],
        "known_destination_url": str(reverse.get("url") or ""),
    }
    target = blocked["target_name"] or blocked["element_id"] or "that control"
    context = (
        f" (context: {blocked['target_context']})"
        if blocked.get("target_context") else ""
    )
    note = (
        "Observed a repeated two-page navigation cycle. On this exact page, "
        f"{blocked['verb']} [{blocked.get('element_id') or ''}] '{target}'{context} previously "
        f"returned to {blocked['known_destination_url'][:90]} and closed the loop. "
        "Do NOT use that same element again. Its label is ambiguous, so inspect its "
        "surrounding card/context and try a different element, provider, or route."
    )
    return NavigationCycleSignal(True, note, blocked)


def navigation_cycle_action_violation(
    state: dict[str, Any], action: Any
) -> str:
    """Reject only the learned loop-closing element, not same-labelled siblings."""
    blocked = state.get("navigation_cycle_blocked_action") or {}
    if not blocked:
        return ""
    verb = getattr(action, "action_type", None) or (
        action.get("verb") if isinstance(action, dict) else ""
    )
    element_id = getattr(action, "element_id", None) if not isinstance(action, dict) else action.get("element_id")
    action_x = getattr(action, "x", None) if not isinstance(action, dict) else action.get("x")
    action_y = getattr(action, "y", None) if not isinstance(action, dict) else action.get("y")
    if str(verb or "") != str(blocked.get("verb") or ""):
        return ""
    blocked_element = blocked.get("element_id")
    if blocked_element and element_id != blocked_element:
        blocked_context = str(blocked.get("target_context") or "").strip().casefold()
        current_context = ""
        if element_id:
            element = (state.get("selector_map") or {}).get(element_id, {}) or {}
            current_context = str(
                element.get("hint") or element.get("container") or ""
            ).strip().casefold()
        context_matches = bool(blocked_context and current_context == blocked_context)
        coordinate_matches = False
        if not element_id and action_x is not None and action_y is not None:
            blocked_live = (state.get("selector_map") or {}).get(blocked_element, {}) or {}
            try:
                coordinate_matches = (
                    abs(float(action_x) - float(blocked_live.get("x"))) <= 24
                    and abs(float(action_y) - float(blocked_live.get("y"))) <= 24
                )
            except (TypeError, ValueError):
                coordinate_matches = False
        if not context_matches and not coordinate_matches:
            return ""
    if not blocked_element:
        target = ""
        if isinstance(action, dict):
            target = str(action.get("target_name") or "")
        if target.strip().casefold() != str(blocked.get("target_name") or "").strip().casefold():
            return ""
    return str(state.get("navigation_cycle_note") or "Known navigation-cycle action must not be repeated.")


def _has_action_cycle(signatures: list[str]) -> bool:
    """Detect a short repeating cycle in the recent action signatures.

    Catches A,A,A (length-1 cycle) and A,B,A,B (length-2 cycle) — the classic
    'alternate between two useless actions' churn that exact-repeat guards miss.
    """
    sigs = [s for s in (signatures or []) if s]
    if len(sigs) < 4:
        return False
    tail = sigs[-6:]
    # length-1: last 3 identical
    if len(set(tail[-3:])) == 1:
        return True
    # length-2: last 4 form X,Y,X,Y with X != Y
    last4 = tail[-4:]
    if len(last4) == 4 and last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
        return True
    return False


def detect_stagnation(state: dict) -> StagnationSignal:
    """Combine the progress signals into a single stagnation verdict (pure)."""
    from cognition import detect_stall

    reasons: list[str] = []
    recent_survey_progress = bool(state.get("recent_survey_progress"))

    same_url = int(state.get("same_url_streak", 0) or 0)
    url_stuck = same_url >= URL_STUCK_THRESHOLD and not recent_survey_progress
    if url_stuck:
        reasons.append(f"on the same URL for {same_url} steps")

    goal_flat = (
        detect_stall(state.get("goal_score_window", []) or [])
        and not recent_survey_progress
    )
    if goal_flat:
        reasons.append("goal-progress score is flat")

    cycle = _has_action_cycle(state.get("loop_signatures", []) or [])
    if cycle:
        reasons.append("repeating a short action cycle")

    level = sum((url_stuck, goal_flat, cycle))
    # A confirmed A,B,A,B action cycle is already sufficient evidence of churn,
    # even if every leg changes URL and superficially resembles survey progress.
    stuck = cycle or level >= 2
    note = ""
    if stuck:
        note = (
            "You are NOT making real progress toward the goal ("
            + "; ".join(reasons)
            + "). Stop repeating variations of the same approach. Re-read the page, "
            "and either act on a clearly DIFFERENT element that advances the goal, "
            "scroll to reveal something new, or re-navigate — do not loop."
        )
    return StagnationSignal(stuck=stuck, level=level, reasons=reasons, note=note)
