"""Deterministic worker fast paths; these must remain model-free."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("workers.deterministic")


def _remove_human_assistance_action(decision):
    """Replace a legacy human-help proposal with autonomous re-perception."""
    if str(decision.action_type or "").lower() != "ask_user":
        return decision
    return decision.model_copy(update={
        "action_type": "wait", "wait_ms": 1000,
        "needs_vision": True,
        "vision_question": "Resolve the current interaction using current visual and DOM evidence.",
        "reasoning": "Autonomous recovery: re-perceive the controls and try a different grounded action.",
        "expected_change": "Fresh perception exposes a grounded autonomous action or confirms the unchanged-page timeout.",
    })
def _survey_fast_path(
    state: dict[str, Any], unavailable_offer_ids: set[str]
) -> dict[str, Any] | None:
    """Resolve mechanically certain survey states before spending a model call."""
    if not state.get("continuous_survey_mode"):
        return None
    selector_map = state.get("selector_map", {}) or {}
    page_text = str(state.get("page_text") or "")
    if not selector_map:
        return None
    recovery_active = bool(
        int(state.get("stagnation_level", 0) or 0) > 0
        or int(state.get("consecutive_identical_actions", 0) or 0) >= 2
        or re.search(
            r"\b(?:loop|repeated|stagnat|ineffective|no effect)\b",
            str(state.get("correction_context") or ""), re.IGNORECASE,
        )
    )
    try:
        from agent_first_browse.survey.context import (
            blocking_popup_action_id,
            is_image_code_page,
            preferred_forward_control_id,
            prepare_survey_transaction,
            rank_survey_offers,
            survey_offer_selection_route,
            paidwork_selection_ready,
            survey_gate_violation,
            survey_visible_form_completeness,
        )

        # Qmee keeps an active-survey marker server-side even after every local
        # survey/dashboard tab has been closed.  Resolve both stages of its
        # conflict page from exact live labels before generic popup handling or
        # another model call.
        try:
            from agent_first_browse.survey.site_quirks import qmee_active_survey_action
            conflict_id, conflict_stage = qmee_active_survey_action(
                str(state.get("current_url") or ""),
                page_text,
                selector_map,
                previous_boundary=str(state.get("survey_last_boundary_outcome") or ""),
            )
            if conflict_id:
                target = selector_map[conflict_id]
                return {
                    "verb": "click", "element_id": conflict_id,
                    "target_name": str(target.get("text") or target.get("name") or "Qmee survey conflict")[:100],
                    "target_context": str(target.get("hint") or "")[:120],
                    "text": None, "url": None, "x": None, "y": None,
                    "rationale": "Deterministic Qmee active-survey recovery.",
                    "reasoning": (
                        "Continue with the newly selected survey and clear Qmee's stale active-survey marker "
                        f"using the truthful prior-boundary status ({conflict_stage})."
                    ),
                    "expected_change": "Qmee clears the prior active-survey state and opens the new survey.",
                    "risk_level": "REVERSIBLE", "reversible": True,
                    "question_text": "Qmee active survey conflict",
                    "answer_basis": "page_navigation", "queued_actions": [],
                    "execution_mode": "single_action",
                }
        except Exception as exc:
            logger.debug("Qmee active-survey fast path skipped (non-fatal): %s", exc)

        popup_id = blocking_popup_action_id(selector_map)
        if popup_id:
            target = selector_map[popup_id]
            return {
                "verb": "click", "element_id": popup_id,
                "target_name": str(target.get("text") or target.get("name") or "close popup")[:100],
                "target_context": str(target.get("hint") or "")[:120],
                "text": None, "url": None, "x": None, "y": None,
                "rationale": "Deterministic popup-first fast path.",
                "reasoning": "Resolve the blocking modal before touching the page behind it.",
                "expected_change": "The blocking modal disappears.",
                "risk_level": "REVERSIBLE", "reversible": True,
                "question_text": "Blocking popup", "answer_basis": "page_navigation",
                "queued_actions": [], "execution_mode": "single_action",
            }

        offers = []
        current_url = str(state.get("current_url") or "")
        if paidwork_selection_ready(current_url, page_text, selector_map) is False:
            waits = int(state.get("paidwork_selection_waits", 0) or 0)
            if waits >= 3:
                # The provider shell has failed to produce either cards or a
                # definitive empty state. Leave the nested provider route once
                # and let the next fresh dashboard perception start cleanly.
                recovery_url = (
                    "https://www.paidwork.com/earn"
                    if current_url.rstrip("/").lower().endswith("/earn/filling-out")
                    else "https://www.paidwork.com/earn/filling-out"
                )
                logger.warning(
                    "🛟 Paidwork selection unavailable after %d waits; recovering to %s",
                    waits, recovery_url,
                )
                return {
                    "verb": "goto", "element_id": None, "text": None,
                    "url": recovery_url, "wait_ms": None,
                    "target_name": "Paidwork survey selection recovery",
                    "question_text": "Paidwork survey selection unavailable",
                    "answer_basis": "page_navigation", "queued_actions": [],
                    "execution_mode": "single_action", "risk_level": "REVERSIBLE",
                    "reversible": True,
                    "rationale": "Bounded provider loading recovery.",
                    "reasoning": "The provider did not expose cards or an empty state within the bounded wait budget.",
                    "expected_change": "A fresh Paidwork survey-selection route loads.",
                }
            return {
                "verb": "wait", "element_id": None, "text": None, "url": None,
                "wait_ms": 1200, "target_name": "Paidwork survey list loading",
                "question_text": "Paidwork survey selection is still loading",
                "answer_basis": "page_navigation", "queued_actions": [],
                "execution_mode": "single_action", "risk_level": "REVERSIBLE",
                "reversible": True,
                "rationale": "Wait for the provider's survey cards or explicit empty state.",
                "reasoning": "The Paidwork shell is present but no survey selection evidence is available yet.",
                "expected_change": "Survey cards or a definitive no-surveys message appears.",
            }
        if survey_offer_selection_route(current_url):
            offers = [
                offer for offer in rank_survey_offers(selector_map)
                if offer.element_id not in unavailable_offer_ids
            ]
        if offers:
            best = offers[0]
            target = selector_map.get(best.element_id, {})
            return {
                "verb": "click", "element_id": best.element_id,
                "target_name": str(target.get("text") or target.get("name") or best.text)[:100],
                "target_context": str(target.get("hint") or "")[:120],
                "text": None, "url": None, "x": None, "y": None,
                "rationale": "Deterministic reward-per-minute fast path.",
                "reasoning": "Choose the highest currently available reward-to-time survey offer.",
                "expected_change": "The selected survey opens or begins loading.",
                "risk_level": "REVERSIBLE", "reversible": True,
                "question_text": "Survey offer dashboard", "answer_basis": "reward_per_minute",
                "offer_reward": str(best.reward), "offer_minutes": float(best.minutes),
                "offer_currency": best.currency,
                "queued_actions": [], "execution_mode": "single_action",
            }

        try:
            from agent_first_browse.survey.site_quirks import provider_start_action
            provider_action_id, provider_stage = provider_start_action(
                str(state.get("current_url") or ""), page_text, selector_map
            )
            if provider_action_id:
                target = selector_map[provider_action_id]
                return {
                    "verb": "click", "element_id": provider_action_id,
                    "target_name": str(target.get("text") or target.get("name") or provider_stage)[:100],
                    "target_context": str(target.get("hint") or "")[:120],
                    "text": None, "url": None, "x": None, "y": None,
                    "rationale": "Deterministic provider survey-entry sequence.",
                    "reasoning": f"Advance the reviewed provider entry stage: {provider_stage}.",
                    "expected_change": "The provider opens the next survey-entry stage or external questionnaire.",
                    "risk_level": "REVERSIBLE", "reversible": True,
                    "question_text": "Provider survey entry", "answer_basis": "page_navigation",
                    "queued_actions": [], "execution_mode": "single_action",
                }
        except Exception as exc:
            logger.debug("Provider entry fast path skipped (non-fatal): %s", exc)

        # Treat a DOB widget as one atomic profile operation. This supports
        # native date inputs, separate day/month/year controls, and providers
        # that hide those controls behind an "alternative calendar" link.
        try:
            from agent_first_browse.survey.profile import load_active_profile, profile_date_of_birth_action
            profile = load_active_profile() or (state.get("survey_profile", {}) or {})
            dob_action = profile_date_of_birth_action(profile, selector_map, page_text)
            if dob_action:
                return dob_action
        except Exception as exc:
            logger.debug("DOB fast path skipped (non-fatal): %s", exc)

        try:
            from agent_first_browse.survey.profile import load_active_profile, profile_native_select_action
            profile = load_active_profile() or (state.get("survey_profile", {}) or {})
            select_action = profile_native_select_action(profile, selector_map, page_text)
            if select_action:
                return select_action
        except Exception as exc:
            logger.debug("Profile select fast path skipped (non-fatal): %s", exc)

        try:
            from agent_first_browse.survey.profile import load_active_profile
            from agent_first_browse.survey.recipes import get_survey_recipe_memory
            recipe_action = None if (
                recovery_active or not survey_offer_selection_route(current_url)
            ) else get_survey_recipe_memory().recall(
                url=str(state.get("current_url") or ""),
                page_text=page_text,
                selector_map=selector_map,
                profile=load_active_profile() or (state.get("survey_profile", {}) or {}),
            )
            if recipe_action:
                recipe_queue, _recipe_reason = prepare_survey_transaction(
                    recipe_action, [], selector_map,
                    page_text=page_text, continuous_mode=True,
                )
                recipe_action["queued_actions"] = recipe_queue
                recipe_action["execution_mode"] = (
                    "page_transaction" if recipe_queue else "single_action"
                )
                return recipe_action
        except Exception as exc:
            logger.debug("Survey recipe recall skipped (non-fatal): %s", exc)

        if not is_image_code_page(page_text):
            forward_id = preferred_forward_control_id(selector_map)
            form_state = survey_visible_form_completeness(selector_map, page_text)
            if forward_id and form_state["has_answer"] and form_state["complete"]:
                action = {
                    "verb": "click", "element_id": forward_id, "text": None,
                    "target_name": str(selector_map[forward_id].get("text") or "Next")[:100],
                    "question_text": page_text[:300], "answer_basis": "page_navigation",
                }
                if not survey_gate_violation(
                    action, selector_map, page_text=page_text, continuous_mode=True
                ):
                    return {
                        **action, "target_context": "", "url": None, "x": None, "y": None,
                        "rationale": "Current answer is already present; advance without another model call.",
                        "reasoning": "The live form-state gate confirms this question is answered.",
                        "expected_change": "A different survey question or route appears with no validation error.",
                        "risk_level": "REVERSIBLE", "reversible": True,
                        "queued_actions": [], "execution_mode": "single_action",
                    }

        # Recognized demographic inputs are deterministic profile lookups. When
        # several are on one page, prepare one guarded human-typing transaction.
        if not is_image_code_page(page_text) and not recovery_active:
            from agent_first_browse.survey.profile import enforce_typed_profile_fact, load_active_profile
            profile = load_active_profile() or (state.get("survey_profile", {}) or {})
            recognized: list[dict[str, Any]] = []
            text_inputs = [
                (element_id, element) for element_id, element in selector_map.items()
                if str(element.get("kind") or "").lower() in {"input", "textarea"}
                and str(element.get("tag") or "").lower() != "select"
                and str(element.get("control_type") or "").lower()
                not in {
                    "button", "submit", "radio", "checkbox", "hidden",
                    "select", "select-one",
                }
            ]
            for element_id, element in text_inputs:
                candidate = {
                    "verb": "type", "element_id": str(element_id), "text": "",
                    "target_name": str(element.get("text") or element.get("name") or "")[:100],
                    "target_context": str(element.get("hint") or "")[:120],
                    "question_text": page_text[:300], "answer_basis": "configured_profile_fact",
                }
                guarded, note, violation = enforce_typed_profile_fact(
                    candidate, profile, selector_map, page_text=page_text
                )
                if note and not violation and guarded.get("text"):
                    current = str(element.get("value") or "").strip()
                    if not current or guarded.get("force_retype"):
                        recognized.append(guarded)
            # On a multi-input form, a broad page heading can describe only one
            # field. If it made several targets resolve to the same profile key,
            # decline the fast path and let the worker map labels individually.
            recognized_keys = [str(item.get("profile_update_key") or "") for item in recognized]
            if len(text_inputs) > 1 and len(set(recognized_keys)) != len(recognized_keys):
                recognized = []
            if recognized:
                primary, remainder = recognized[0], recognized[1:]
                queue, _reason = prepare_survey_transaction(
                    primary, remainder, selector_map,
                    page_text=page_text, continuous_mode=True,
                )
                return {
                    **primary, "url": None, "x": None, "y": None,
                    "rationale": "Use the authoritative profile for a recognized factual field.",
                    "reasoning": "This field has one exact configured profile value.",
                    "expected_change": "The field contains the profile value and passes local validation.",
                    "risk_level": "REVERSIBLE", "reversible": True,
                    "queued_actions": queue,
                    "execution_mode": "page_transaction" if queue else "single_action",
                }
    except Exception as exc:
        logger.debug("Deterministic survey fast path skipped (non-fatal): %s", exc)
    return None

