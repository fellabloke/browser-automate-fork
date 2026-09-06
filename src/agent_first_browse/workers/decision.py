"""Semantic worker decision orchestration and model-backed invocation.

The implementation is intentionally preserved from the former worker module;
this module only establishes ownership and keeps the existing call contract.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_first_browse.logging import get_logger
from agent_first_browse.workers.deterministic import (
    _remove_human_assistance_action,
    _survey_fast_path,
)
from agent_first_browse.workers.prompt_builder import survey_focus_instructions
from agent_first_browse.workers.schemas import WorkerAction

logger = get_logger("workers")


_SUPPORTED_WORKER_ACTIONS = {
    "goto", "click", "type", "scroll", "press_enter", "wait", "done",
    "hover", "select_option", "press_key", "press_combo", "drag_and_drop",
    "upload_file", "scroll_to", "set_date_of_birth", "abandon_survey",
}


try:
    WEB_DREAMER_TIMEOUT_SECONDS = max(
        1.0, float(os.getenv("WEB_DREAMER_TIMEOUT_SECONDS", "20"))
    )
except (TypeError, ValueError):
    WEB_DREAMER_TIMEOUT_SECONDS = 20.0


def _prompt_char_limit(name: str, default: int, minimum: int = 300) -> int:
    """Read one defensive prompt-size cap without letting a bad env break a run."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _bounded_prompt_section(value: Any, max_chars: int, *, recent: bool = False) -> str:
    """Compact a prompt section while retaining useful boundaries.

    History is useful from the newest end.  DOM/handoff/profile sections need
    both their headers and their latest controls, so they retain head + tail.
    """
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    marker = "\n… bounded prompt content omitted …\n"
    if recent:
        recent_marker = marker.lstrip()
        usable = max(1, max_chars - len(recent_marker))
        return recent_marker + text[-usable:]
    usable = max(1, max_chars - len(marker))
    head = int(usable * 0.6)
    return text[:head] + marker + text[-(usable - head):]


def _worker_prompt_limit() -> int:
    """Final user-message cap before a provider call."""
    return _prompt_char_limit("SURVEY_WORKER_PROMPT_MAX_CHARS", 8000, minimum=4000)
async def _validate_coord_click(
    invoke_fn, vision_chain, breaker, health_tracker,
    *, x: float, y: float, intended_target: str, objective: str,
) -> tuple[bool, str]:
    """Ask Vision to verify what is at (x, y) before clicking.

    This is the "Look-Before-You-Leap" safety net for current coordinate fallback.
    When Vision returns pixel coordinates instead of an element_id (because the
    target is absent from the a11y tree), we take a SECOND screenshot and ask
    Vision to confirm that the coordinate actually matches the intended target.

    Returns (valid, reason). Does NOT modify Overwatch — if valid, the action
    flows to Overwatch with coordinates that pass its grounding check naturally.
    """
    if invoke_fn is None or not vision_chain:
        return False, "no vision chain available"

    from agent_first_browse.actions.tools import mcp_screenshot
    shot = await mcp_screenshot(full_page=False)
    if not shot.get("ok"):
        return False, "screenshot failed for coord validation"

    from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

    class _CoordValidation(BaseModel):
        element_at_point: str = Field(
            description="Describe EXACTLY what visual element/button/link is at the marked coordinate."
        )
        matches_target: bool = Field(
            description="Does the element at the coordinate match the intended target?"
        )
        confidence: float = Field(description="0.0-1.0 confidence in the match.")
        reasoning: str = Field(description="Why this is or is not the right target.")

    vp_w = shot.get("width", 1440)
    vp_h = shot.get("height", 1080)
    messages = [
        _SM(content=(
            "You are a precision visual validator for a browser automation agent. "
            "You will be given a screenshot, a coordinate point [x,y], and an "
            "intended target description. Your ONLY job: look at EXACTLY what is "
            "at that coordinate on the screenshot and determine if it matches the "
            "intended target. Be precise — a 'Star' button is NOT the same as a "
            "'5 stars' text link. A 'Fork' button is NOT the same as a fork count."
        )),
        _HM(content=(
            f"═══ COORDINATE TO VALIDATE ═══\n"
            f"Point: ({x:.0f}, {y:.0f}) on a {vp_w}×{vp_h} viewport\n\n"
            f"═══ INTENDED TARGET ═══\n{intended_target}\n\n"
            f"═══ OBJECTIVE ═══\n{objective}\n\n"
            f"Look at the screenshot. What EXACTLY is at coordinate "
            f"({x:.0f}, {y:.0f})? Is it '{intended_target}'?"
        )),
    ]

    try:
        from agent_first_browse.perception.vision import (
            VISION_FAILOVER_BUDGET_SECONDS,
            VISION_MODEL_TIMEOUT_SECONDS,
            VISION_TIMEOUT_COOLDOWN_SECONDS,
        )

        result, _model = await invoke_fn(
            vision_chain, messages, _CoordValidation,
            breaker, base64_image=shot["base64"], health_tracker=health_tracker,
            timeout_seconds=VISION_MODEL_TIMEOUT_SECONDS,
            total_timeout_seconds=VISION_FAILOVER_BUDGET_SECONDS,
            timeout_cooldown_seconds=VISION_TIMEOUT_COOLDOWN_SECONDS,
        )
        if result is None:
            return False, "vision validation returned None"
        if result.matches_target and result.confidence >= 0.65:
            logger.info(
                "✅ COORD VALIDATED: (%.0f,%.0f) confirmed as '%s' [%.0f%%]",
                x, y, result.element_at_point[:50], result.confidence * 100,
            )
            return True, f"confirmed: {result.element_at_point[:60]}"
        else:
            logger.warning(
                "❌ COORD REJECTED: (%.0f,%.0f) is '%s', not '%s' [%.0f%%]",
                x, y, result.element_at_point[:50], intended_target[:40],
                result.confidence * 100,
            )
            return False, f"mismatch: found '{result.element_at_point[:60]}'"
    except Exception as e:
        logger.warning("coord validation error: %s", str(e)[:120])
        return False, f"validation error: {str(e)[:80]}"
async def invoke_worker(
    state: dict[str, Any],
    system_prompt: str,
    failover_chain: list,
    breaker,
    health_tracker,
    invoke_fn,
    vision_chain: list | None = None,
    dreamer=None,
) -> dict[str, Any]:
    """Core worker invocation logic shared by all specialists.

    Args:
        state: Current BrainState dict
        system_prompt: Specialist-specific system prompt
        failover_chain: LLM failover chain
        breaker: CircuitBreaker instance
        health_tracker: ProviderHealthTracker instance
        invoke_fn: The _invoke_with_failover function

    Returns:
        State updates dict with proposed_action, etc.
    """
    # ── Build User Prompt ──
    qmee_visual_recovery = False
    observer_recovery_used = False
    objective = state.get("objective", "")
    current_url = state.get("current_url", "")
    step_number = state.get("step_number", 0)
    max_steps = state.get("max_steps", 25)
    dom_markdown = state.get("dom_markdown", "")
    page_text = state.get("page_text", "")
    login_detected = state.get("login_detected", False)
    correction_context = state.get("correction_context", "")
    recovery_advice = state.get("recovery_advice", "")
    consecutive_identical = state.get("consecutive_identical_actions", 0)
    plan_render = state.get("plan_render", "")
    facts_render = state.get("facts_render", "")
    history_compressed = state.get("history_compressed", "")
    skill_context = state.get("skill_context", "")
    survey_profile_render = state.get("survey_profile_render", "")
    survey_cycle_memory_render = state.get("survey_cycle_memory_render", "")
    model_wait_seconds = 0.0

    async def timed_invoke(*args, **kwargs):
        """Measure model time so survey inactivity excludes inference latency."""
        nonlocal model_wait_seconds
        started = time.monotonic()
        try:
            return await invoke_fn(*args, **kwargs)
        finally:
            model_wait_seconds += max(0.0, time.monotonic() - started)

    try:
        from agent_first_browse.survey.context import build_survey_handoff
        survey_handoff = build_survey_handoff(state)
    except Exception as e:  # noqa: BLE001
        logger.debug("Survey handoff skipped (non-fatal): %s", e)
        survey_handoff = ""

    survey_unavailable_offer_ids: set[str] = set()
    if survey_handoff:
        try:
            from agent_first_browse.survey.context import recently_failed_survey_offer_ids
            survey_unavailable_offer_ids = recently_failed_survey_offer_ids(
                state.get("selector_map", {}) or {},
                list(state.get("history") or []),
                current_url=current_url,
            )
            if survey_unavailable_offer_ids:
                logger.info(
                    "🚫 Temporarily skipping failed survey offer(s): %s",
                    ", ".join(sorted(survey_unavailable_offer_ids)),
                )
            remembered_unsupported = {
                re.sub(r"\s+", " ", str(value or "")).strip().lower()
                for value in (state.get("survey_unsupported_offer_signatures") or [])
                if str(value or "").strip()
            }
            if remembered_unsupported:
                from agent_first_browse.survey.context import rank_survey_offers
                for offer in rank_survey_offers(state.get("selector_map", {}) or {}):
                    if re.sub(r"\s+", " ", offer.text).strip().lower() in remembered_unsupported:
                        survey_unavailable_offer_ids.add(offer.element_id)
                if survey_unavailable_offer_ids:
                    logger.info(
                        "🚫 Skipping previously unsupported survey offer(s): %s",
                        ", ".join(sorted(survey_unavailable_offer_ids)),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Survey offer quarantine skipped (non-fatal): %s", exc)

    # Survey prompts previously repeated the full rendered page inside the
    # handoff and then again below it, while carrying a large DOM and history.
    # Keep the authoritative pieces, but place a deterministic ceiling on each
    # section so one long survey cannot exceed free-provider TPM limits.
    prompt_survey_handoff = survey_handoff
    prompt_cycle_memory = survey_cycle_memory_render
    prompt_history = history_compressed
    prompt_dom = dom_markdown
    prompt_profile = survey_profile_render
    prompt_current_url = current_url
    if survey_handoff:
        try:
            from agent_first_browse.survey.context import compact_survey_url
            prompt_current_url = compact_survey_url(current_url)
        except Exception:
            prompt_current_url = current_url[:360]
        prompt_survey_handoff = _bounded_prompt_section(
            survey_handoff,
            _prompt_char_limit("SURVEY_WORKER_HANDOFF_MAX_CHARS", 3000),
        )
        prompt_cycle_memory = _bounded_prompt_section(
            survey_cycle_memory_render,
            _prompt_char_limit("SURVEY_WORKER_CYCLE_MEMORY_MAX_CHARS", 1000),
            recent=True,
        )
        prompt_history = _bounded_prompt_section(
            history_compressed,
            _prompt_char_limit("SURVEY_WORKER_HISTORY_MAX_CHARS", 1200),
            recent=True,
        )
        prompt_dom = _bounded_prompt_section(
            dom_markdown,
            _prompt_char_limit("SURVEY_WORKER_DOM_MAX_CHARS", 5000),
        )
        prompt_profile = _bounded_prompt_section(
            survey_profile_render,
            _prompt_char_limit("SURVEY_WORKER_PROFILE_MAX_CHARS", 3000),
        )

    # Perception owns terminal survey-boundary decisions. They are based on the
    # verified submission ledger and the exact wall-clock page identity, so do
    # not spend another model call asking whether to leave or risk the model
    # overriding the deterministic timeout/provider policy.
    boundary_target = str(state.get("survey_boundary_target_url") or "").strip()
    boundary_reason = str(state.get("survey_boundary_reason") or "").strip()
    if (
        state.get("continuous_survey_mode")
        and state.get("survey_abandon_required")
        and boundary_target
    ):
        logger.warning(
            "↩️ Deterministic survey boundary (%s) → %s",
            boundary_reason or "abandoned",
            boundary_target[:100],
        )
        return {
            "proposed_action": {
                "verb": "abandon_survey",
                "element_id": None,
                "target_name": "survey provider dashboard",
                "text": None,
                "url": boundary_target,
                "x": None,
                "y": None,
                "rationale": "Close the terminal/stuck survey and continue autonomously.",
                "risk_level": "REVERSIBLE",
                "reversible": True,
                "question_text": "",
                "answer_basis": "page_navigation",
                "survey_boundary_reason": boundary_reason or "abandoned",
            },
            "last_action_signature": f"abandon_survey:{boundary_target[:80]}",
            "consecutive_identical_actions": 0,
        }

    if survey_handoff:
        # Overwatch can reject a model action after a Qmee tab/popup handoff
        # because the dashboard regenerated every eN handle.  Force one fresh
        # perception before the fast path/model sees that stale snapshot.
        if re.search(r"qmee\.com/.*/surveys|qmee\.com/en-gb/surveys", current_url, re.I):
            latest_history = (state.get("history") or [])[-1:]
            latest_text = " ".join(
                str(latest.get(key) or "")
                for latest in latest_history
                for key in ("action_outcome", "outcome", "reasoning")
            ).lower()
            latest_item = latest_history[0] if latest_history else {}
            latest_id = str(latest_item.get("element_id") or "")
            latest_verb = str(latest_item.get("verb") or latest_item.get("action_type") or "").lower()
            stale_id_after_handoff = (
                latest_verb in {"click", "select_option", "type", "press", "scroll"}
                and bool(latest_id)
                and latest_id not in (state.get("selector_map") or {})
            )
            if (
                "not in current snapshot" in latest_text
                or "grounding reject" in latest_text
                or stale_id_after_handoff
            ):
                return {
                    "proposed_action": {
                        "verb": "wait", "element_id": None, "text": None, "url": None,
                        "wait_ms": 700, "target_name": "Qmee dashboard refresh",
                        "question_text": "Survey dashboard loading",
                        "answer_basis": "page_navigation", "queued_actions": [],
                        "execution_mode": "single_action", "risk_level": "REVERSIBLE",
                        "reversible": True,
                        "rationale": "Refresh the live Qmee dashboard after a stale target rejection.",
                        "reasoning": "Do not replay an element ID that Overwatch confirmed is absent.",
                        "expected_change": "A fresh selector map with live survey offers appears.",
                    },
                    "last_action_signature": "wait:qmee-dashboard-refresh",
                    "consecutive_identical_actions": 0,
                }
        fast_action = _survey_fast_path(state, survey_unavailable_offer_ids)
        if fast_action:
            try:
                from agent_first_browse.survey.context import survey_ineffective_action_violation
                fast_violation = survey_ineffective_action_violation(state, fast_action)
                if fast_violation:
                    logger.info("⚡ Fast path quarantined: %s", fast_violation[:140])
                    fast_action = None
            except Exception:
                pass
        if fast_action:
            fast_action["snapshot_revision"] = str(state.get("snapshot_revision") or "")
            logger.info(
                "⚡ DETERMINISTIC SURVEY FAST PATH: %s [%s]",
                fast_action.get("verb"), fast_action.get("element_id") or "",
            )
            signature = (
                f"{fast_action.get('verb')}:"
                f"{fast_action.get('element_id') or (fast_action.get('text') or 'none')[:30]}"
            )
            return {
                "proposed_action": fast_action,
                "last_action_signature": signature,
                "consecutive_identical_actions": (
                    int(state.get("consecutive_identical_actions", 0) or 0) + 1
                    if signature == state.get("last_action_signature") else 0
                ),
                "survey_hold_identity": str(fast_action.get("held_action_identity") or state.get("survey_hold_identity") or ""),
                "survey_hold_count": int(fast_action.get("hold_count") or state.get("survey_hold_count", 0) or 0),
                "survey_gate_exhausted": bool(
                    fast_action.get("survey_gate_hold")
                    and int(fast_action.get("hold_count", 0) or 0) >= 2
                ),
            }

        # A refreshed Qmee dashboard can briefly contain only stale/partial
        # accessibility data after a popup or tab handoff. Do not let the model
        # replay dead eN handles from its previous snapshot; wait for one fresh
        # map and let the next pass rank live offers again.
        if re.search(r"qmee\.com/.*/surveys|qmee\.com/en-gb/surveys", current_url, re.I):
            from agent_first_browse.survey.context import rank_survey_offers
            if not any(
                str(item.get("element_id") or "") in (state.get("selector_map") or {})
                for item in (state.get("history") or [])[-2:]
            ) and not rank_survey_offers(state.get("selector_map", {}) or {}):
                # One short wait lets a freshly restored SPA hydrate. A second
                # identical wait previously looped forever and never allowed
                # the worker or observer to inspect a Qmee popup.
                if state.get("last_action_signature") == "wait:qmee-dashboard-refresh":
                    qmee_visual_recovery = True
                else:
                    return {
                        "proposed_action": {
                            "verb": "wait", "element_id": None, "text": None, "url": None,
                            "wait_ms": 700, "target_name": "Qmee dashboard refresh",
                            "question_text": "Survey dashboard loading",
                            "answer_basis": "page_navigation", "queued_actions": [],
                            "execution_mode": "single_action", "risk_level": "REVERSIBLE",
                            "reversible": True,
                            "rationale": "Fresh dashboard perception required after popup/tab refresh.",
                            "reasoning": "Do not replay stale dashboard element IDs.",
                            "expected_change": "A fresh live survey offer map appears.",
                        },
                        "last_action_signature": "wait:qmee-dashboard-refresh",
                        "consecutive_identical_actions": 0,
                    }

    # ── current Cognition: the agent reasons WITH its persistent strategy + beliefs ──
    # strategy block is the PERSISTENT context only; the goal_complete_hint
    # (transient "finish now" nudge) is now owned by the Guidance Bus below.
    from agent_first_browse.cognition.reasoning import render_strategy_block, build_guidance
    strategy_block = render_strategy_block(
        strategy=state.get("strategy", ""),
        confidence=state.get("strategy_confidence", 1.0),
        beliefs=state.get("beliefs", []) or [],
        success_criteria=state.get("success_criteria", ""),
    )

    # ── current Target Lock: bind this step to the target item's semantic identity so
    #    identical-looking distractor controls (a neighbor's 'Add to cart') can
    #    never steal focus — even when the agent is confused. ──
    bound_target = None
    target_lock_block = ""
    try:
        from agent_first_browse.config.feature_flags import target_lock_enabled
        if target_lock_enabled():
            from agent_first_browse.cognition.target_lock import extract_target, render_target_lock_block
            active_sub = ""
            for _s in state.get("plan_steps", []) or []:
                if _s.get("status") in ("active", "in_progress"):
                    active_sub = _s.get("desc", "")
                    break
            bound_target = extract_target(objective, active_sub)
            target_lock_block = render_target_lock_block(bound_target, objective)
    except Exception as e:  # noqa: BLE001
        logger.debug("Target Lock skipped (non-fatal): %s", e)

    # ── current Atomic Intent Journal: if the PREVIOUS side-effecting action did not
    #    return a confirmed success, surface the pending-action ledger here so the
    #    worker — and EVERY model in the failover chain that answers this same
    #    prompt — is warned not to blindly repeat it (handoff-amnesia fix). ──
    pending_intent = state.get("last_attempted_action")
    hesitation_block = ""
    try:
        from agent_first_browse.config.feature_flags import intent_journal_enabled
        if intent_journal_enabled() and pending_intent:
            from agent_first_browse.memory.intent_journal import render_hesitation
            hesitation_block = render_hesitation(pending_intent)
            if hesitation_block:
                logger.info("⚠️ HESITATION: predecessor %s on '%s' unconfirmed — "
                            "warning worker not to blindly repeat",
                            pending_intent.get("verb"),
                            pending_intent.get("target_name") or pending_intent.get("element_id") or "")
    except Exception as e:  # noqa: BLE001
        logger.debug("Hesitation block skipped (non-fatal): %s", e)

    # ── current Sub-Goal Lock: surface the FORBID-list of already-completed (verified)
    #    sub-goals so the agent never re-does finished work — even after a global
    #    'done' rejection. This forbids; it adds no pending focus, so it is
    #    complementary to plan_steps (no competing-checklist regression). ──
    lock_list_block = ""
    try:
        from agent_first_browse.config.feature_flags import subgoal_lock_enabled
        if subgoal_lock_enabled() and state.get("prm_checklist"):
            from agent_first_browse.cognition.subgoal_lock import render_lock_list
            lock_list_block = render_lock_list(state.get("prm_checklist"))
    except Exception as e:  # noqa: BLE001
        logger.debug("Lock-list block skipped (non-fatal): %s", e)

    # Login hint (informational, non-conflicting — not part of the guidance bus)
    login_hint = ""
    if login_detected:
        login_hint = (
            "\n\n🔑 LOGIN STATUS: You appear to be LOGGED IN. "
            "Profile/account indicators detected. "
            "DO NOT attempt to log in again."
        )

    # ── current Guidance Bus: exactly ONE arbitrated transient directive ──
    # Replaces the old stacking of correction_context + recovery_advice +
    # dedup_override + critical_action_hint (the F2 regression). The PRM-checklist
    # re-injection is GONE — plan_steps (system prompt) is the only sub-goal source.
    guidance_block = build_guidance(state)
    if guidance_block:
        logger.info("🧭 GUIDANCE: %s", guidance_block.split("\n", 1)[-1][:120])

    user_prompt = (
        f"═══ YOUR MISSION ═══\n"
        f"{objective}\n\n"
        f"NOTE: The above describes the user's INTENT. Any numbered steps are guidance, not strict commands. "
        f"YOU decide the actual actions based on what you see on screen.\n\n"
        + (f"{strategy_block}\n"
           "Follow your strategy, but ADAPT it to what you actually see. When your "
           "DONE WHEN criteria are met, output action_type='done' — do not keep acting "
           "after the goal is achieved.\n\n" if strategy_block else "")
        + (f"{target_lock_block}\n\n" if target_lock_block else "")
        + f"═══ CURRENT CONTEXT ═══\n"
        f"URL: {prompt_current_url}\n"
        f"Automation action turn: {step_number+1}/{max_steps} "
        f"({max_steps - step_number - 1} action turns remaining; NOT survey progress)"
        f"{login_hint}\n\n"
        + (f"{prompt_survey_handoff}\n\n" if prompt_survey_handoff else "")
        + (f"{prompt_cycle_memory}\n\n" if prompt_cycle_memory else "")
        + (f"{guidance_block}\n\n" if guidance_block else "")
        + (f"{hesitation_block}\n\n" if hesitation_block else "")
        + (f"{lock_list_block}\n\n" if lock_list_block else "")
        + f"═══ ACTION HISTORY (compressed) ═══\n"
        f"{prompt_history or '(first step)'}\n\n"
        # The authoritative survey handoff already carries current page text;
        # do not pay for the same content twice on every survey decision.
        + (f"═══ CURRENT RENDERED PAGE TEXT ═══\n{page_text[:4000]}\n\n"
           if page_text and not survey_handoff else "")
        + f"═══ PAGE STRUCTURE ═══\n"
        f"{prompt_dom}\n\n"
        # Objective anchor at the BOTTOM of the prompt (sandwich pattern).
        # On pages with large DOM markdown, the objective at the top gets buried
        # in the context window. Repeating it here ensures the LLM's attention
        # stays locked on the goal, not on pattern-matching DOM elements.
        f"═══ REMEMBER YOUR MISSION ═══\n"
        f"{objective[:300]}\n"
        f"Now OBSERVE the page structure above, THINK about your situation, and choose your NEXT action."
    )

    # Per-section caps still allow strategy and correction text to push the
    # aggregate request over a provider's context/TPM limit. Keep the mission
    # anchor at the head and the current DOM/reminder at the tail.
    prompt_limit = _worker_prompt_limit()
    if len(user_prompt) > prompt_limit:
        user_prompt = _bounded_prompt_section(user_prompt, prompt_limit)
        logger.info("✂️ Worker prompt compacted to %d chars", len(user_prompt))

    runtime_system_prompt = (
        system_prompt + survey_focus_instructions(objective) +
        "\n\nOPERATING STYLE: You are a calm, evidence-led browser operator. "
        "Observe the current screen, choose one concrete next action, predict its "
        "visible result, and verify it. Never repeat an ineffective action; if the "
        "screen is unclear or progress stalls, request a fresh vision consult and "
        "change the tactic."
    )
    if prompt_profile and survey_handoff:
        runtime_system_prompt += "\n\n" + prompt_profile
    system_prompt_limit = _prompt_char_limit(
        "SURVEY_WORKER_SYSTEM_PROMPT_MAX_CHARS", 8000, minimum=5000
    )
    if survey_handoff and len(runtime_system_prompt) > system_prompt_limit:
        runtime_system_prompt = _bounded_prompt_section(
            runtime_system_prompt, system_prompt_limit
        )
        logger.info(
            "✂️ Worker system prompt compacted to %d chars",
            len(runtime_system_prompt),
        )
    messages = [SystemMessage(content=runtime_system_prompt), HumanMessage(content=user_prompt)]
    worker_invoke_kwargs: dict[str, Any] = {}
    if survey_handoff:
        try:
            worker_invoke_kwargs = {
                "timeout_seconds": max(
                    2.0, float(os.getenv("SURVEY_WORKER_MODEL_TIMEOUT_SECONDS", "15"))
                ),
                "total_timeout_seconds": max(
                    4.0, float(os.getenv("SURVEY_WORKER_FAILOVER_BUDGET_SECONDS", "45"))
                ),
                "timeout_cooldown_seconds": max(
                    30.0, float(os.getenv("SURVEY_MODEL_TIMEOUT_COOLDOWN_SECONDS", "60"))
                ),
                "timeout_sibling_threshold": 1,
                "role": "TEXT_WORKER",
            }
        except (TypeError, ValueError):
            worker_invoke_kwargs = {
                "timeout_seconds": 15.0,
                "total_timeout_seconds": 45.0,
                "timeout_cooldown_seconds": 60.0,
                "timeout_sibling_threshold": 1,
                "role": "TEXT_WORKER",
            }

    # ── Invoke LLM ──
    try:
        decision, used_model = await timed_invoke(
            failover_chain, messages, WorkerAction,
            breaker, health_tracker=health_tracker,
            **worker_invoke_kwargs,
        )
        logger.info("Worker answered by: %s", used_model)
    except RuntimeError as e:
        wait_secs = 0.5 if survey_handoff else 5.0
        # If the text worker is unavailable while the browser is already stuck,
        # give the visual observer a chance to recover the run. Previously this
        # returned {} and the graph simply repeated perception with no new eyes.
        recovery_needed = bool(
            vision_chain and survey_handoff and (
                state.get("correction_context")
                or int(state.get("ineffective_streak", 0) or 0) >= 1
                or int(state.get("consecutive_identical_actions", 0) or 0) >= 1
            )
        )
        if recovery_needed:
            from agent_first_browse.perception.vision import consult_vision, apply_vision_verdict
            logger.warning("👁️ Text worker unavailable while stalled; asking the observer why progress stopped")
            observer_verdict, observer_model = await consult_vision(
                timed_invoke, vision_chain, breaker, health_tracker,
                objective=objective,
                question=(
                    "Diagnose why the survey cannot progress. Inspect the entire popup and viewport, "
                    "including the obvious large Next arrow at the bottom. Explain what was completed, "
                    "what is blocking progress, and identify one concrete next action."
                ),
                a11y_markdown=dom_markdown,
                history_tail=history_compressed,
                allow_cache=False,
            )
            if observer_verdict:
                recovered = {
                    "verb": observer_verdict.action_type,
                    "element_id": observer_verdict.element_id,
                    "x": None,
                    "y": None,
                    "target_element_id": observer_verdict.target_element_id,
                    "target_x": observer_verdict.target_x,
                    "target_y": observer_verdict.target_y,
                    "text": observer_verdict.text,
                    "reasoning": observer_verdict.reasoning,
                }
                recovered, overridden = apply_vision_verdict(recovered, observer_verdict)
                if overridden and recovered.get("verb") not in {"none", "wait"}:
                    decision = WorkerAction.model_validate({
                        "screen_state": observer_verdict.scene_summary or observer_verdict.observation,
                        "previous_action_result": history_compressed[-500:],
                        "goal_progress": "Observer recovery after stalled text reasoning.",
                        "question_text": observer_verdict.target_description or observer_verdict.next_step,
                        "answer_basis": "unknown_needs_vision",
                        "reasoning": recovered.get("reasoning", ""),
                        "expected_change": "The visible target is acted upon and the survey advances.",
                        "action_type": recovered["verb"],
                        "element_id": recovered.get("element_id"),
                        "x": recovered.get("x"), "y": recovered.get("y"),
                        "target_element_id": recovered.get("target_element_id"),
                        "target_x": recovered.get("target_x"),
                        "target_y": recovered.get("target_y"),
                        "text": recovered.get("text"),
                        "needs_vision": False,
                        "vision_question": observer_verdict.blockage,
                    })
                    used_model = f"vision-recovery:{observer_model}"
                    observer_recovery_used = True
                    logger.info("👁️ Observer recovered stalled action: %s [%s] — %s",
                                decision.action_type, decision.element_id or "coords",
                                observer_verdict.next_step[:160])
                else:
                    logger.warning("👁️ Observer could not produce a grounded recovery action")
                    await asyncio.sleep(wait_secs)
                    return {}
            else:
                logger.warning("👁️ Observer unavailable during stalled-worker recovery")
                await asyncio.sleep(wait_secs)
                return {}
        else:
            logger.error("Worker LLM FAILURE: %s — waiting %.1fs", e, wait_secs)
            await asyncio.sleep(wait_secs)
            return {}  # Empty update — step will be retried

    # Survey runs are autonomous. Drag challenges
    # often use canvas/custom elements omitted from accessibility snapshots, so
    # recover their visible geometry directly and produce a real drag action.
    if survey_handoff and state.get("continuous_survey_mode"):
        drag_match = re.search(
            r"drag\s+and\s+drop(?:\s+the)?(?:\s+number)?\s+([a-z0-9]+)",
            str(state.get("page_text") or ""), re.IGNORECASE,
        )
        if drag_match:
            try:
                from agent_first_browse.actions.tools import mcp_find_drag_targets
                geometry = await mcp_find_drag_targets(drag_match.group(1))
                if geometry.get("ok"):
                    decision = decision.model_copy(update={
                        "action_type": "drag_and_drop", "element_id": None,
                        "x": geometry["source_x"], "y": geometry["source_y"],
                        "target_x": geometry["target_x"], "target_y": geometry["target_y"],
                        "needs_vision": False, "answer_basis": "attention_instruction",
                        "reasoning": "Drag the requested visible item into the detected square drop zone.",
                        "expected_change": "The requested item moves into the square and Next becomes enabled.",
                    })
                    logger.info("🧲 Grounded visual drag geometry for '%s'", drag_match.group(1))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Drag geometry recovery failed (non-fatal): %s", exc)

    if qmee_visual_recovery:
        decision = decision.model_copy(update={
            "needs_vision": True,
            "vision_question": (
                "Qmee still has no usable survey offers after a hydration wait. "
                "Inspect the full viewport for a popup, modal, cookie banner, "
                "active-survey warning, or visually rendered offer and choose "
                "one concrete grounded action."
            ),
        })

    # Defense in depth for stale provider responses or an old cached schema.
    # Human assistance is not an executable action in this runtime. Convert it
    # globally to bounded autonomous re-perception and replace every narrative
    # field so downstream logs/prompts cannot continue expecting a person.
    if str(decision.action_type or "").lower() == "ask_user":
        decision = _remove_human_assistance_action(decision)
        logger.warning("♾️ Removed unsupported human-assistance proposal; continuing autonomously")

    normalized_action = str(decision.action_type or "").strip().lower()
    if normalized_action not in _SUPPORTED_WORKER_ACTIONS:
        logger.warning(
            "🧰 Worker proposed unsupported action '%s'; forcing visual re-perception",
            normalized_action or "(empty)",
        )
        decision = decision.model_copy(update={
            "action_type": "wait",
            "wait_ms": 500,
            "element_id": None,
            "needs_vision": True,
            "vision_question": (
                f"The text worker requested unsupported tool '{normalized_action}'. "
                "Identify the current obstacle and choose one supported grounded action."
            ),
            "reasoning": "Unsupported tool request; re-observe visually before acting.",
            "expected_change": "Fresh visual evidence identifies a supported action.",
        })

    # Provider failure pages are not ordinary survey questions. In continuous
    # mode leave them deterministically and return to the last known offer list;
    # asking another model what to do here often causes repeated retries.
    if survey_handoff and state.get("continuous_survey_mode"):
        try:
            from agent_first_browse.survey.context import survey_failure_kind
            failure_kind = survey_failure_kind(state.get("page_text", ""))
            home_url = str(state.get("survey_home_url") or "").strip()
            current_url = str(state.get("current_url") or "")
            empty_survey_page = (
                not str(state.get("page_text") or "").strip()
                and not (state.get("selector_map") or {})
                and current_url != home_url
                and int(state.get("survey_empty_page_streak", 0) or 0) >= 2
            )
            requested_home_return = bool(
                decision.action_type == "goto"
                and home_url
                and str(decision.url or "").split("?", 1)[0].rstrip("/")
                == home_url.split("?", 1)[0].rstrip("/")
                and current_url.split("?", 1)[0].rstrip("/")
                != home_url.split("?", 1)[0].rstrip("/")
            )
            timed_out = bool(state.get("survey_stuck_timed_out"))
            if (failure_kind or empty_survey_page or timed_out) and home_url and home_url != current_url:
                failure_kind = failure_kind or "load_failed"
                decision = decision.model_copy(update={
                    "action_type": "abandon_survey", "url": home_url, "element_id": None,
                    "x": None, "y": None, "text": None,
                    "question_text": f"Survey provider failure ({failure_kind}); return to offer list",
                    "answer_basis": "page_navigation",
                    "reasoning": "Leave failed/disqualified survey and select the next best-value offer.",
                    "expected_change": "The active survey tab closes and a fresh provider dashboard opens.",
                })
                logger.warning("↩️ Survey %s detected; returning to offer list %s",
                               failure_kind, home_url[:100])
            elif requested_home_return:
                decision = decision.model_copy(update={
                    "action_type": "wait", "url": None, "element_id": None,
                    "x": None, "y": None, "wait_ms": 1000,
                    "needs_vision": True,
                    "vision_question": "Find a grounded way to complete the current survey interaction.",
                    "reasoning": "A plain dashboard navigation would leave the survey tab/session open; wait for the verified boundary.",
                    "expected_change": "Fresh perception finds an autonomous action or the unchanged-page timer reaches its close boundary.",
                })
                logger.warning("↩️ Rejected plain survey-dashboard goto; boundary cleanup must close the tab")
        except Exception as e:  # noqa: BLE001
            logger.debug("Survey failure recovery skipped (non-fatal): %s", e)

    # Page-authored text can influence any input, not just a textarea. Resolve
    # recognized factual fields from the durable profile before the first survey
    # gate so copied trap phrases never trigger a needless model retry.
    typed_profile_guard_reason = ""
    if survey_handoff and decision.action_type == "type":
        try:
            from agent_first_browse.survey.profile import enforce_typed_profile_fact, load_active_profile
            guarded, profile_note, typed_profile_guard_reason = enforce_typed_profile_fact(
                decision.model_dump(),
                load_active_profile() or (state.get("survey_profile", {}) or {}),
                state.get("selector_map", {}) or {},
                page_text=state.get("page_text", ""),
            )
            if profile_note:
                decision = WorkerAction.model_validate(guarded)
                logger.info("🛡️ TYPED PROFILE GUARD: %s", profile_note)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Initial typed profile guard skipped (non-fatal): %s", exc)

    if survey_handoff and decision.action_type in {"click", "select_option"}:
        try:
            from agent_first_browse.survey.profile import enforce_profile_choice, load_active_profile
            guarded, profile_note, profile_choice_violation = enforce_profile_choice(
                decision.model_dump(),
                load_active_profile() or (state.get("survey_profile", {}) or {}),
                state.get("selector_map", {}) or {},
                page_text=state.get("page_text", ""),
            )
            if profile_note:
                decision = WorkerAction.model_validate(guarded)
                logger.info("🛡️ PROFILE CHOICE GUARD: %s", profile_note)
            if profile_choice_violation:
                typed_profile_guard_reason = profile_choice_violation
        except Exception as exc:  # noqa: BLE001
            logger.debug("Initial profile choice guard skipped (non-fatal): %s", exc)

    # An observed A→B→A→B outcome is stronger evidence than an ambiguous label.
    # Give the worker one immediate re-decision with the exact loop-closing
    # element forbidden, while leaving same-labelled siblings selectable.
    navigation_cycle_gate_reason = ""
    try:
        from agent_first_browse.cognition.stagnation import navigation_cycle_action_violation
        navigation_cycle_gate_reason = navigation_cycle_action_violation(state, decision)
        if navigation_cycle_gate_reason:
            logger.warning(
                "🔂 NAVIGATION CYCLE rejected %s [%s]: %s",
                decision.action_type,
                decision.element_id or "",
                navigation_cycle_gate_reason[:160],
            )
            correction = HumanMessage(content=(
                "OBSERVED ACTION-EFFECT LOOP — YOUR PROPOSAL IS FORBIDDEN:\n"
                f"{navigation_cycle_gate_reason}\n\n"
                "The label was misleading. Choose a GENUINELY DIFFERENT CURRENT "
                "element_id or route. Inspect each control's surrounding card/provider "
                "context; a sibling with the same label is allowed only when it has a "
                "different element_id. Do not propose the blocked element again."
            ))
            corrected, corrected_model = await timed_invoke(
                failover_chain, messages + [correction], WorkerAction,
                breaker, health_tracker=health_tracker,
                **worker_invoke_kwargs,
            )
            decision, used_model = corrected, corrected_model
            navigation_cycle_gate_reason = navigation_cycle_action_violation(state, corrected)
            logger.info("Worker navigation-cycle correction answered by: %s", corrected_model)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Navigation-cycle correction skipped (non-fatal): %s", exc)

    # Deterministic survey gate: a model (especially a fallback) may choose Next
    # merely because it is prominent, or re-click an already-selected radio. Give
    # the ensemble one immediate correction using the SAME current-state prompt.
    survey_gate_reason = ""
    if survey_handoff:
        try:
            from agent_first_browse.survey.context import (
                popup_blocks_action,
                preferred_survey_offer_id,
                survey_gate_violation,
            )
            popup_action = popup_blocks_action(
                decision, state.get("selector_map", {}) or {}
            )
            if popup_action and popup_action != decision.element_id:
                blocked_target = decision.element_id
                decision.element_id = popup_action
                decision.answer_basis = "page_navigation"
                decision.reasoning = (
                    f"Deterministic popup-first router redirected the blocked target "
                    f"[{blocked_target}] to modal action [{popup_action}]. " + decision.reasoning
                )[:1000]
                logger.info(
                    "🪟 POPUP-FIRST ROUTER: [%s] → [%s] without another model call",
                    blocked_target,
                    popup_action,
                )
            preferred_offer = preferred_survey_offer_id(
                decision,
                state.get("selector_map", {}) or {},
                survey_unavailable_offer_ids,
            )
            if preferred_offer and preferred_offer != decision.element_id:
                original_offer = decision.element_id
                decision.element_id = preferred_offer
                decision.reasoning = (
                    f"Deterministic live value ranking redirected [{original_offer}] to "
                    f"[{preferred_offer}]. " + decision.reasoning
                )[:1000]
                logger.info(
                    "🧮 SURVEY VALUE ROUTER: [%s] → [%s] without another model call",
                    original_offer,
                    preferred_offer,
                )
            survey_gate_reason = typed_profile_guard_reason or survey_gate_violation(
                decision,
                state.get("selector_map", {}) or {},
                page_text=state.get("page_text", ""),
                audio_analysis=state.get("survey_audio_analysis") or {},
                continuous_mode=bool(state.get("continuous_survey_mode")),
                unavailable_offer_ids=survey_unavailable_offer_ids,
            )
            if survey_gate_reason and any(phrase in survey_gate_reason.lower() for phrase in (
                "already contains the proposed answer",
                "proposed option is already selected",
            )):
                from agent_first_browse.survey.context import preferred_forward_control_id
                forward_id = preferred_forward_control_id(
                    state.get("selector_map", {}) or {}
                )
                forward_action = {
                    "verb": "click", "element_id": forward_id,
                    "text": None, "answer_basis": "page_navigation",
                }
                forward_violation = survey_gate_violation(
                    forward_action,
                    state.get("selector_map", {}) or {},
                    page_text=state.get("page_text", ""),
                    continuous_mode=bool(state.get("continuous_survey_mode")),
                ) if forward_id else survey_gate_reason
                if forward_id and not forward_violation:
                    decision = decision.model_copy(update={
                        "action_type": "click", "element_id": forward_id,
                        "text": None, "answer_basis": "page_navigation",
                        "reasoning": "The answer is already present; advance without retyping/reselecting.",
                        "expected_change": "The survey advances to another question or validation.",
                        "queued_actions": [],
                    })
                    survey_gate_reason = ""
                    logger.info(
                        "🧷 Duplicate-answer gate redirected to [%s] without a correction model call",
                        forward_id,
                    )
            if survey_gate_reason:
                logger.warning("🧷 SURVEY GATE rejected %s [%s]: %s",
                               decision.action_type, decision.element_id or "",
                               survey_gate_reason[:140])
                correction = HumanMessage(content=(
                    "SURVEY ACTION GATE REJECTED YOUR PROPOSAL:\n"
                    f"{survey_gate_reason}\n\n"
                    "Re-read the CURRENT rendered question/instructions and CURRENT "
                    "element map. Return a corrected action. State the question in "
                    "question_text and classify answer_basis. Attention instructions "
                    "must be followed literally; do not default to the first option."
                ))
                corrected, corrected_model = await timed_invoke(
                    failover_chain, messages + [correction], WorkerAction,
                    breaker, health_tracker=health_tracker,
                    **worker_invoke_kwargs,
                )
                corrected_profile_violation = ""
                if corrected.action_type == "type":
                    try:
                        from agent_first_browse.survey.profile import enforce_typed_profile_fact, load_active_profile
                        guarded, profile_note, corrected_profile_violation = enforce_typed_profile_fact(
                            corrected.model_dump(),
                            load_active_profile() or (state.get("survey_profile", {}) or {}),
                            state.get("selector_map", {}) or {},
                            page_text=state.get("page_text", ""),
                        )
                        if profile_note:
                            corrected = WorkerAction.model_validate(guarded)
                            logger.info("🛡️ TYPED PROFILE GUARD after correction: %s", profile_note)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Corrected typed profile guard skipped (non-fatal): %s", exc)
                corrected_violation = survey_gate_violation(
                    corrected,
                    state.get("selector_map", {}) or {},
                    page_text=state.get("page_text", ""),
                    audio_analysis=state.get("survey_audio_analysis") or {},
                    continuous_mode=bool(state.get("continuous_survey_mode")),
                    unavailable_offer_ids=survey_unavailable_offer_ids,
                )
                corrected_violation = corrected_profile_violation or corrected_violation
                decision, used_model = corrected, corrected_model
                survey_gate_reason = corrected_violation
                logger.info("Worker survey correction answered by: %s", corrected_model)
                if corrected_violation:
                    # Vision gets one chance to read a difficult attention item.
                    # If unavailable, a later guard converts this to wait rather
                    # than submitting an unanswered question.
                    decision.needs_vision = True
                    decision.vision_question = corrected_violation
        except Exception as e:  # noqa: BLE001
            logger.debug("Survey action gate correction skipped (non-fatal): %s", e)

    # ── Log OTA chain ──
    logger.info("👁️ OBSERVE: %s", decision.screen_state[:200])
    logger.info("🔄 PREVIOUS: %s", decision.previous_action_result[:150])
    logger.info("📊 PROGRESS: %s", decision.goal_progress[:150])
    logger.info("❓ QUESTION: %s", decision.question_text[:180])
    logger.info("🧾 ANSWER BASIS: %s", decision.answer_basis[:120])
    logger.info("🧠 REASONING: %s", decision.reasoning[:200])
    if getattr(decision, "expected_change", ""):
        logger.info("🔮 EXPECT: %s", decision.expected_change[:180])
    logger.info("⚡ ACTION: %s", decision.action_type)

    # ── Classify action risk ──
    from agent_first_browse.cognition.action_classifier import classify_action, ActionRisk
    target_name = ""
    target_context = ""
    selector_map = state.get("selector_map", {})
    if decision.element_id and decision.element_id in selector_map:
        el_data = selector_map[decision.element_id]
        target_name = el_data.get("name", el_data.get("text", ""))[:60]
        target_context = str(el_data.get("hint") or el_data.get("container") or "")[:120]

    risk = classify_action(
        action_type=decision.action_type,
        target_name=target_name,
        target_text=decision.text[:60] if decision.text else "",
        url=current_url,
        element_kind=selector_map.get(decision.element_id, {}).get("kind", "") if decision.element_id else "",
    )

    # ── Build ProposedAction ──
    proposed = {
        "verb": decision.action_type,
        "element_id": decision.element_id,
        "target_name": target_name,
        "target_context": target_context,
        "text": decision.text,
        "key_combo": decision.key_combo,
        "file_path": decision.file_path,
        "direction": decision.direction,
        "scroll_amount": decision.scroll_amount,
        "url": decision.url,
        "x": decision.x,
        "y": decision.y,
        "rationale": decision.reasoning[:200],
        "risk_level": risk.name,
        "reversible": risk == ActionRisk.REVERSIBLE,
        "screen_state": decision.screen_state[:200],
        "previous_action_result": decision.previous_action_result[:150],
        "goal_progress": decision.goal_progress[:150],
        "question_text": decision.question_text[:300],
        "answer_basis": decision.answer_basis[:80],
        "profile_update_category": decision.profile_update_category[:40],
        "profile_update_key": decision.profile_update_key[:80],
        "profile_update_mode": decision.profile_update_mode[:20],
        "profile_update_value": decision.profile_update_value[:160],
        "profile_update_reason": decision.profile_update_reason[:250],
        "reasoning": decision.reasoning[:200],
        "expected_change": getattr(decision, "expected_change", "")[:250],
        "wait_ms": decision.wait_ms,
        "target_x": getattr(decision, "target_x", None),
        "target_y": getattr(decision, "target_y", None),
        "target_element_id": getattr(decision, "target_element_id", None),
        "snapshot_revision": str(state.get("snapshot_revision") or ""),
        "queued_actions": [item.model_dump() for item in decision.queued_actions[:8]],
    }
    queue_anchor = (
        proposed.get("verb"), proposed.get("element_id"), proposed.get("text")
    )

    grounded_survey_choice = False
    if survey_handoff:
        try:
            from agent_first_browse.survey.context import is_grounded_survey_choice
            grounded_survey_choice = is_grounded_survey_choice(proposed, selector_map)
        except Exception:  # noqa: BLE001
            pass

    # ══════════════════════════════════════════════════════════════════════
    #  P2 DYNAMIC CASCADE CONSENSUS — need-based, latency-aware
    #  IRREVERSIBLE actions can't be retried, so they get a second opinion — but
    #  via a CASCADE, not an always-on N-voter poll (latency is critical):
    #    Tier 1  primary confident + structurally sound → execute NOW (0 extra calls)
    #    Tier 2  else poll the SECONDARY; if it agrees → execute       (1 extra call)
    #    Tier 3  else poll the TERTIARY → CISC vote                    (2 extra calls)
    #    Abstain vote still split → vision / safe-wait
    #  All voting lives in consensus.cascade_consensus (modular — this block only
    #  triggers it and applies the result; it never touches prompt/guidance state).
    # ══════════════════════════════════════════════════════════════════════
    consensus_update: dict[str, Any] = {}
    force_consult = False

    # ── current Clarity Gate: how clear/unambiguous is this action? (drives BOTH the
    #    broadened pre-action consensus and the vision trigger below). ──
    clarity_sig = None
    try:
        from agent_first_browse.cognition.clarity import compute_clarity
        clarity_input = {
            **proposed,
            "confidence": float(getattr(decision, "confidence", 0.7) or 0.7),
            "needs_vision": bool(getattr(decision, "needs_vision", False)),
        }
        clarity_sig = compute_clarity(clarity_input, state, target=bound_target,
                                      selector_map=selector_map)
        if clarity_sig.uncertain:
            logger.info("🔎 CLARITY: uncertain (%s)", "; ".join(clarity_sig.reasons[:3]))
    except Exception as e:  # noqa: BLE001
        logger.debug("Clarity signal skipped (non-fatal): %s", e)

    # Decide the objective vision triggers before polling the text ensemble.
    # If vision is already inevitable (explicit worker request, escalation rung,
    # or repeated ineffective actions), text consensus cannot remove that need;
    # paying for both stages only delays the same screenshot consult.
    from agent_first_browse.perception.vision import should_consult_vision
    vision_trigger_state = state
    if observer_recovery_used:
        # The fallback action was just derived from a fresh screenshot. Do not
        # spend another vision call merely because the pre-existing ineffective
        # streak is still present in graph state.
        vision_trigger_state = {
            **state,
            "ineffective_streak": 0,
            "force_vision": False,
        }
    preconsult, preconsult_reason = should_consult_vision(
        needs_vision=getattr(decision, "needs_vision", False),
        state=vision_trigger_state,
        action_type=decision.action_type,
    )
    from agent_first_browse.survey.context import (
        captcha_field_state,
        captcha_refresh_id,
        is_image_code_page,
    )
    captcha_page = is_image_code_page(state.get("page_text", ""))
    if captcha_page:
        _captcha_input_id, _captcha_filled = captcha_field_state(selector_map)
        _captcha_refresh = captcha_refresh_id(selector_map)
        if (
            decision.action_type == "click"
            and decision.element_id == _captcha_refresh
            and not _captcha_filled
        ):
            preconsult, preconsult_reason = False, "explicit CAPTCHA refresh does not need vision"
        else:
            preconsult = True
            if _captcha_filled:
                preconsult_reason = (
                    "Independently read the CAPTCHA image again and compare it character-for-character, "
                    "including case, with the filled input. Click the enabled forward control only if "
                    "they exactly match; type the corrected code on mismatch; use Refresh if uncertain."
                )
            else:
                preconsult_reason = (
                    "Read the exact CAPTCHA characters, preserving case. Type only a confident complete "
                    "code; use Refresh rather than guessing if any character is uncertain."
                )
    if not preconsult and clarity_sig is not None:
        try:
            from agent_first_browse.perception.vision import MAX_VISION_CONSULTS
            from agent_first_browse.cognition.clarity import needs_vision_for_clarity

            cl_v, cl_why = needs_vision_for_clarity(clarity_sig)
            if cl_v and int(state.get("vision_consults", 0) or 0) < MAX_VISION_CONSULTS:
                preconsult, preconsult_reason = True, cl_why
        except Exception:  # noqa: BLE001
            pass

    # Navigation/consent controls that are present in the live selector map do
    # not benefit from a screenshot. Suppress model-requested or clarity-driven
    # vision here; otherwise a fallback can repeatedly escalate the same obvious
    # button and exhaust every vision provider before executing it.
    dom_grounded_navigation = (
        decision.action_type in {"click", "type"}
        and decision.element_id in selector_map
        and str(getattr(decision, "answer_basis", "")).strip().lower() == "page_navigation"
        and not proposed.get("vision_coords")
    )
    # A required consult outranks the cheap DOM shortcut. Otherwise a stuck
    # navigation action can suppress the screenshot intended to disambiguate it.
    vision_is_forced = bool(
        preconsult
        or force_consult
        or getattr(decision, "needs_vision", False)
        or state.get("force_vision")
        or int(state.get("ineffective_streak", 0) or 0) >= 1
    )
    dom_grounded_navigation = dom_grounded_navigation and not vision_is_forced
    if dom_grounded_navigation:
        if preconsult:
            logger.info("👁️ Vision suppressed: DOM-grounded page-navigation action [%s]",
                        decision.element_id)
        preconsult, preconsult_reason = False, "DOM-grounded page navigation"

    try:
        from agent_first_browse.cognition.consensus import (consensus_enabled, count_distinct_base_models,
                               cascade_consensus)
        from agent_first_browse.cognition.clarity import needs_consensus
        from agent_first_browse.config.feature_flags import clarity_consensus_enabled

        is_irrev = (risk == ActionRisk.IRREVERSIBLE)
        if clarity_sig is not None:
            do_vote, vote_reason = needs_consensus(
                clarity_sig, is_irreversible=is_irrev,
                broaden=clarity_consensus_enabled())
        else:
            do_vote, vote_reason = is_irrev, "irreversible action"

        # current Intent Journal: re-proposing an UNCONFIRMED prior action is the
        # double-toggle risk — never fire it blind; force a second opinion first.
        repeating_uncertain = False
        try:
            from agent_first_browse.memory.intent_journal import same_action
            if pending_intent and same_action(pending_intent, proposed):
                repeating_uncertain = True
                do_vote = True
                vote_reason = ((vote_reason + "; ") if vote_reason else "") + \
                    "repeating an UNCONFIRMED prior action (possible double-apply)"
        except Exception:  # noqa: BLE001
            pass

        routine_grounded_survey_action = bool(
            survey_handoff
            and risk == ActionRisk.REVERSIBLE
            and decision.action_type in {"click", "type", "select_option", "press_key"}
            and (
                decision.element_id in selector_map
                or decision.action_type == "press_key"
            )
            and str(decision.answer_basis or "").lower()
            not in {"unknown_needs_vision", ""}
            and not captcha_page
            and not preconsult
            and not repeating_uncertain
        )
        if routine_grounded_survey_action and do_vote:
            logger.info("🗳️ Consensus suppressed for reversible DOM-grounded survey action")
            do_vote = False
            vote_reason = "routine reversible survey action"

        if (do_vote and consensus_enabled()
                and not (preconsult and vision_chain)
                and count_distinct_base_models(failover_chain) >= 2):
            # PRE-action: we get the second opinion BEFORE the action is executed.
            logger.info("🗳️ PRE-ACTION CONSENSUS — %s", vote_reason)
            # Hesitation / stuck / any uncertainty forces a real second opinion even
            # on a confident primary (the zero-risk rule).
            force_escalate = (state.get("consecutive_identical_actions", 0) >= 2
                              or bool(state.get("correction_context"))
                              or repeating_uncertain
                              or (clarity_sig is not None and clarity_sig.uncertain))
            cascade = await cascade_consensus(
                primary_decision=decision, primary_model=used_model,
                messages=messages, schema=WorkerAction,
                invoke_fn=timed_invoke, chain=failover_chain,
                breaker=breaker, health_tracker=health_tracker,
                selector_map=selector_map, force_escalate=force_escalate)

            consensus_update["consensus_votes"] = state.get("consensus_votes", 0) + 1
            logger.info("🗳️ CASCADE [%s] +%d calls, agreement=%.0f%% — %s",
                        cascade.path, cascade.extra_calls,
                        cascade.agreement * 100, cascade.detail)

            if cascade.abstain:
                consensus_update["abstentions"] = state.get("abstentions", 0) + 1
                if vision_chain:
                    logger.warning("🛑 ABSTAIN → consulting vision to disambiguate "
                                   "the ambiguous action before acting.")
                    force_consult = True
                elif is_irrev:
                    logger.warning("🛑 ABSTAIN (irreversible) → no vision; holding "
                                   "(wait) and re-examining before committing.")
                    proposed = {**proposed, "verb": "wait", "wait_ms": 800,
                                "risk_level": "REVERSIBLE", "reversible": True}
                    consensus_update["correction_context"] = (
                        "\n\n⚖️ The critical irreversible action is AMBIGUOUS — the "
                        "models disagreed on the target. Re-examine the page and "
                        "identify the SINGLE correct element before committing.")
                else:
                    # Reversible + no vision: the voters split, but a wrong reversible
                    # action is recoverable via the verifier-gated retry, so proceed
                    # on the primary rather than waste a step waiting.
                    logger.info("⚖️ ABSTAIN (reversible) → proceeding on primary "
                                "(a retry can recover if wrong).")
            else:
                cd = cascade.decision
                if (cd is not decision
                        and (getattr(cd, "action_type", None) != decision.action_type
                             or getattr(cd, "element_id", None) != decision.element_id)):
                    # The cascade settled on a DIFFERENT action than the primary pick.
                    win_el = selector_map.get(cd.element_id, {}) if cd.element_id else {}
                    win_target = (win_el.get("name", win_el.get("text", "")) or "")[:60]
                    win_risk = classify_action(
                        action_type=cd.action_type, target_name=win_target,
                        target_text=(cd.text or "")[:60], url=current_url,
                        element_kind=win_el.get("kind", ""))
                    logger.info("🗳️ CASCADE override → %s [%s]",
                                cd.action_type, cd.element_id or "")
                    proposed = {**proposed, "verb": cd.action_type,
                                "element_id": cd.element_id, "text": cd.text,
                                "target_x": cd.target_x, "target_y": cd.target_y,
                                "target_element_id": cd.target_element_id,
                                "target_name": win_target, "risk_level": win_risk.name,
                                "reversible": win_risk == ActionRisk.REVERSIBLE,
                                "question_text": cd.question_text[:300],
                                "answer_basis": cd.answer_basis[:80],
                                "profile_update_category": cd.profile_update_category[:40],
                                "profile_update_key": cd.profile_update_key[:80],
                                "profile_update_mode": cd.profile_update_mode[:20],
                                "profile_update_value": cd.profile_update_value[:160],
                                "profile_update_reason": cd.profile_update_reason[:250]}
        elif do_vote and preconsult and vision_chain:
            logger.info("🗳️ Consensus skipped — vision already required (%s)",
                        preconsult_reason)
    except Exception as e:  # noqa: BLE001 — consensus never breaks the step
        logger.debug("Cascade consensus skipped (non-fatal): %s", e)

    # ── Vision-on-demand (current): the agent works on the a11y DOM by default and
    #    "opens its eyes" for ONE step only when it cannot resolve the page from
    #    text — then this update reverts (force_vision cleared) and the next step
    #    is text-only again. ──
    vision_update: dict[str, Any] = {}
    vision_resolved = False
    from agent_first_browse.perception.vision import consult_vision, apply_vision_verdict
    consult, why = preconsult, preconsult_reason
    if force_consult and not dom_grounded_navigation:
        consult, why = True, "consensus abstention — disambiguate critical action"
    vision_attempted = bool(consult and vision_chain)
    # Always perform the cheap live-DOM check before paying for vision. A failed
    # type is usually a focus/target-resolution problem, not a visual
    # ambiguity. Give the deterministic executor one immediate retry after the
    # DOM resolver has re-bound the label to its live input. Vision is reserved
    # for a second failure or a genuinely different/DOM-invisible target.
    dom_target_check: dict[str, Any] = {}
    if consult and not captcha_page and proposed.get("verb") in {"type", "click"}:
        try:
            from agent_first_browse.actions.tools import verify_action_target
            dom_target_check = await verify_action_target(
                proposed.get("element_id"), str(proposed.get("verb") or "")
            )
            if dom_target_check.get("ok"):
                logger.info(
                    "🧭 DOM PRIORITY: verified %s target [%s] before vision",
                    proposed.get("verb"), proposed.get("element_id") or "",
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("DOM priority check skipped: %s", exc)
    video_visual_task = bool(re.search(
        r"(?:video|turnstile|hcaptcha|recaptcha).{0,30}(?:captcha|verification)|"
        r"(?:captcha|verification).{0,30}(?:video|turnstile|hcaptcha|recaptcha)",
        f"{state.get('page_text', '')} {objective}", re.I,
    ))
    explicit_visual_requirement = bool(
        captcha_page or video_visual_task
        or getattr(decision, "needs_vision", False)
        or state.get("force_vision")
        or proposed.get("verb") == "drag_and_drop"
    )
    if dom_target_check.get("ok") and not explicit_visual_requirement:
        consult = False
        preconsult_reason = "DOM target verified; visual disambiguation unnecessary"

    # A failed type is usually a focus/target-resolution problem, not a visual
    # ambiguity. Give the deterministic executor one immediate retry after the
    # DOM resolver has re-bound the label to its live input. Vision is reserved
    # for a second failure or a genuinely different target/action.
    previous_intent = state.get("last_attempted_action") or {}
    cheap_type_recovery = bool(
        consult and vision_chain
        and int(state.get("ineffective_streak", 0) or 0) == 1
        and str(proposed.get("verb") or "") == "type"
        and str(previous_intent.get("verb") or "") == "type"
    )
    if cheap_type_recovery:
        logger.info(
            "🧪 DOM TYPE RECOVERY: retrying the live input once before vision "
            "(previous type was unverified)"
        )
        proposed["dom_recovery_retry"] = True
        proposed["reasoning"] = (
            "The previous type was unverified. Re-resolve the associated live "
            "input and retry once before visual escalation."
        )
        consult = False

    if consult and vision_chain:
        logger.info("🧠→👁️ Escalating to vision: %s", why)
        from agent_first_browse.perception.vision import (
            CAPTCHA as VISION_CAPTCHA,
            CLARITY as VISION_CLARITY,
            FORCED_RECOVERY as VISION_FORCED_RECOVERY,
            INEFFECTIVE_RECOVERY as VISION_INEFFECTIVE_RECOVERY,
            NORMAL as VISION_NORMAL,
        )
        if captcha_page:
            consult_reason = VISION_CAPTCHA
        elif int(state.get("ineffective_streak", 0) or 0) >= 1:
            consult_reason = VISION_INEFFECTIVE_RECOVERY
        elif state.get("force_vision"):
            consult_reason = VISION_FORCED_RECOVERY
        elif getattr(decision, "needs_vision", False):
            consult_reason = VISION_CLARITY
        else:
            consult_reason = VISION_NORMAL
        recovery_context = None
        if consult_reason == VISION_INEFFECTIVE_RECOVERY:
            recovery_context = {
                "previous_action": state.get("last_attempted_action") or state.get("last_action_signature", ""),
                "previous_target": proposed.get("target_name", ""),
                "expected_state_change": proposed.get("expected_change", ""),
                "observed_state_identifier": state.get("survey_page_fingerprint", ""),
                "ineffective_action_count": int(state.get("ineffective_streak", 0) or 0),
            }
        verdict, _vm = await consult_vision(
            timed_invoke, vision_chain, breaker, health_tracker,
            objective=objective,
            question=getattr(decision, "vision_question", "") or why,
            a11y_markdown=dom_markdown,
            history_tail=(
                f"{survey_profile_render}\n\n{survey_handoff}\n\n{history_compressed}"
                if survey_handoff else history_compressed
            ),
            # CAPTCHA and explicit uncertainty requests require an independent
            # fresh visual read; reusing a cached verdict can repeat the mistake
            # that triggered escalation.
            allow_cache=not (
                captcha_page
                or bool(getattr(decision, "needs_vision", False))
                or bool(state.get("force_vision"))
                or consult_reason == VISION_INEFFECTIVE_RECOVERY
            ),
            consult_reason=consult_reason,
            recovery_context=recovery_context,
        )
        before_vision_id = proposed.get("element_id")
        before_vision_target = proposed.get("target_name", "")
        if captcha_page:
            from agent_first_browse.survey.context import reconcile_captcha_vision
            proposed, overridden, captcha_note = reconcile_captcha_vision(
                proposed, verdict, selector_map,
                refresh_count=int(state.get("captcha_refreshes", 0) or 0),
                max_refreshes=max(1, int(os.getenv("CAPTCHA_MAX_REFRESHES", "3"))),
            )
            logger.info("🧩 CAPTCHA state machine: %s", captcha_note)
        else:
            proposed, overridden = apply_vision_verdict(proposed, verdict)
        vision_resolved = overridden
        if overridden and (
            not captcha_page
            or proposed.get("verb") == "type"
            or proposed.get("captcha_verified")
        ):
            proposed["vision_verified"] = True
        vision_update["vision_consults"] = state.get("vision_consults", 0) + 1
        if captcha_page:
            transition = str(proposed.get("captcha_transition") or "")
            counters = {
                "captcha_read_attempts": int(state.get("captcha_read_attempts", 0) or 0),
                "captcha_comparison_attempts": int(state.get("captcha_comparison_attempts", 0) or 0),
                "captcha_corrections": int(state.get("captcha_corrections", 0) or 0),
                "captcha_refreshes": int(state.get("captcha_refreshes", 0) or 0),
            }
            if transition == "READ": counters["captcha_read_attempts"] += 1
            elif transition.startswith("COMPARE"): counters["captcha_comparison_attempts"] += 1
            elif transition == "CORRECT_ONCE": counters["captcha_corrections"] += 1
            elif transition == "REFRESH": counters["captcha_refreshes"] += 1
            vision_update.update(counters)
            vision_update["captcha_last_result"] = transition or "UNCERTAIN"
        if (
            verdict
            and str(verdict.action_type or "").lower() in {"", "none"}
            and proposed.get("verb") in {"wait", "none", ""}
        ):
            # A successful visual read that found no executable action must
            # trigger a fresh DOM/screenshot pass, not another identical wait.
            # Keep this bounded by the existing MAX_VISION_CONSULTS budget.
            proposed = {
                **proposed,
                "verb": "wait",
                "element_id": None,
                "x": None,
                "y": None,
                "wait_ms": 300,
                "needs_vision": True,
                "vision_question": (
                    "Reinspect the active viewport after the last visual read. "
                    "Identify one concrete visible control or report a terminal screen-out."
                ),
                "reasoning": (
                    "Vision found no executable action; refresh perception before waiting again."
                ),
                "expected_change": "Fresh perception exposes a grounded action or a terminal boundary.",
            }
            vision_update["correction_context"] = (
                "The previous vision response described the page but returned no action. "
                "Use fresh DOM and screenshot evidence; do not repeat the same wait."
            )
        vision_update["force_vision"] = False  # consumed — revert to a11y DOM
        if overridden:
            # Vision may change e2 → e5. Rebind target and prediction so
            # Overwatch audits the action vision actually chose, not stale
            # text-model fields that would manufacture a contradiction.
            vision_eid = proposed.get("element_id")
            if vision_eid and vision_eid in selector_map:
                vision_el = selector_map[vision_eid]
                vision_target = (vision_el.get("name", vision_el.get("text", "")) or "")[:60]
                proposed["target_name"] = vision_target
                expected = str(proposed.get("expected_change") or "")
                if before_vision_id:
                    expected = expected.replace(before_vision_id, vision_eid)
                if before_vision_target and vision_target:
                    expected = expected.replace(before_vision_target, vision_target)
                proposed["expected_change"] = expected
            logger.info("👁️→⚡ Vision refined the action → %s [%s]",
                        proposed["verb"], proposed.get("element_id") or "")

            # ── current Coordinate Validation: Look-Before-You-Leap ──
            # When Vision returned raw coordinates (element not in a11y tree),
            # verify the target via a second vision check before sending to
            # Overwatch. This ensures coordinate clicks are precise and safe.
            has_source_coords = (
                proposed.get("x") is not None and proposed.get("y") is not None
            )
            has_target_coords = (
                proposed.get("verb") == "drag_and_drop"
                and proposed.get("target_x") is not None
                and proposed.get("target_y") is not None
            )
            if proposed.get("vision_coords") and (has_source_coords or has_target_coords):
                intended = (
                    (proposed.get("reasoning", "")
                     .replace("[vision] ", "")[:120])
                    or proposed.get("target_name", "unknown target")
                )
                try:
                    checks = []
                    if has_source_coords:
                        source_label = (
                            f"draggable source: {intended}"
                            if proposed.get("verb") == "drag_and_drop"
                            else intended
                        )
                        checks.append((
                            proposed["x"], proposed["y"],
                            source_label,
                        ))
                    if has_target_coords:
                        checks.append((
                            proposed["target_x"], proposed["target_y"],
                            "drop destination: the visible target area for this drag",
                        ))
                    results = [await _validate_coord_click(
                        timed_invoke, vision_chain, breaker, health_tracker,
                        x=x, y=y, intended_target=target, objective=objective,
                    ) for x, y, target in checks]
                    valid = all(item[0] for item in results)
                    reason = "; ".join(item[1] for item in results)
                    if valid:
                        proposed["coord_validated"] = True
                        logger.info("👁️✅ Coordinate target(s) validated: %s", reason[:120])
                    else:
                        logger.warning(
                            "👁️❌ Coordinate target rejected: %s — dropping coordinate action",
                            reason[:80],
                        )
                        if has_source_coords:
                            proposed["x"] = None
                            proposed["y"] = None
                        if has_target_coords:
                            proposed["target_x"] = None
                            proposed["target_y"] = None
                        proposed["vision_coords"] = False
                        proposed["coord_validated"] = False
                        vision_resolved = False
                except Exception as val_err:  # noqa: BLE001
                    logger.warning("coord validation failed (non-fatal): %s", val_err)
                    proposed["coord_validated"] = False

    # If both the text correction and vision failed to resolve a deterministic
    # survey-gate violation, hold safely. Never click Next on an unanswered radio
    # question and never re-click an already-selected radio as a fallback.
    if survey_gate_reason and not vision_resolved:
        try:
            from agent_first_browse.survey.context import survey_gate_violation
            if survey_gate_violation(
                proposed,
                selector_map,
                page_text=state.get("page_text", ""),
                audio_analysis=state.get("survey_audio_analysis") or {},
                continuous_mode=bool(state.get("continuous_survey_mode")),
                unavailable_offer_ids=survey_unavailable_offer_ids,
            ):
                logger.warning("🧷 SURVEY GATE holding action — current question state unresolved")
                proposed = {
                    **proposed,
                    "verb": "wait",
                    "element_id": None,
                    "x": None,
                    "y": None,
                    "survey_gate_hold": True,
                    "reasoning": "Survey gate held an unanswered/repeated control for fresh perception.",
                    "expected_change": "",
                    "risk_level": "REVERSIBLE",
                    "reversible": True,
                }
        except Exception as e:  # noqa: BLE001
            logger.debug("Survey hold guard skipped (non-fatal): %s", e)

    # ══════════════════════════════════════════════════════════════════════
    #  B — WebDreamer: look-before-you-leap on HIGH-STAKES AMBIGUOUS
    #  steps. It IMAGINES (LLM world-model — no real browser action) the outcome
    #  of the top-K candidate actions and picks the best. Gated by the Clarity Gate
    #  (only fires when uncertain) AND a cost gate (should_invoke_dreamer:
    #  irreversible / stuck / confused) so it never burns compute on obvious steps.
    # ══════════════════════════════════════════════════════════════════════
    webdreamer_update: dict[str, Any] = {}
    try:
        from agent_first_browse.config.feature_flags import webdreamer_enabled
        if (webdreamer_enabled() and not grounded_survey_choice
                and not vision_resolved and not vision_attempted
                and dreamer is not None and clarity_sig is not None
                and clarity_sig.uncertain):
            from agent_first_browse.cognition.dreamer import (should_invoke_dreamer, should_override_with_dreamer,
                                     CandidateAction)
            if should_invoke_dreamer(
                    element_count=len(selector_map or {}),
                    action_risk_level=proposed.get("risk_level", "REVERSIBLE"),
                    consecutive_no_progress=int(state.get("ineffective_streak", 0) or 0),
                    step_number=int(state.get("step_number", 0) or 0),
                    same_url_streak=int(state.get("same_url_streak", 0) or 0)):
                proposed_ca = CandidateAction(
                    action_type=proposed["verb"], element_id=proposed.get("element_id"),
                    text=proposed.get("text"), url=proposed.get("url"),
                    x=proposed.get("x"), y=proposed.get("y"),
                    reasoning=proposed.get("reasoning", ""))
                logger.info("🌙 WebDreamer: simulating candidates before a high-stakes "
                            "ambiguous action (%s)…", proposed.get("verb"))
                webdreamer_update["webdreamer_runs"] = state.get("webdreamer_runs", 0) + 1
                try:
                    dr = await asyncio.wait_for(
                        dreamer.plan_and_select(
                            dom_markdown=(
                                f"CURRENT RENDERED QUESTION/INSTRUCTIONS:\n{page_text[:3000]}\n\n"
                                f"{dom_markdown}" if survey_handoff else dom_markdown
                            ), objective=objective,
                            plan_context=plan_render, action_history=history_compressed,
                            current_url=current_url, proposed_action=proposed_ca,
                            situation=state),  # situational tuning (state signals only)
                        timeout=WEB_DREAMER_TIMEOUT_SECONDS)
                    best = dr.best_action
                    logger.info("🌙 WebDreamer: best=%s score=%.2f (k=%d)",
                                best.describe(), dr.best_score, dr.candidates_generated)
                    if should_override_with_dreamer(
                            dr.best_score, best.action_type, best.element_id,
                            proposed["verb"], proposed.get("element_id")):
                        win_el = selector_map.get(best.element_id, {}) if best.element_id else {}
                        win_target = (win_el.get("name", win_el.get("text", "")) or "")[:60]
                        win_risk = classify_action(
                            action_type=best.action_type, target_name=win_target,
                            target_text=(best.text or "")[:60], url=current_url,
                            element_kind=win_el.get("kind", ""))
                        proposed = {**proposed, "verb": best.action_type,
                                    "element_id": best.element_id, "text": best.text,
                                    "url": best.url, "target_name": win_target,
                                    "risk_level": win_risk.name,
                                    "reversible": win_risk == ActionRisk.REVERSIBLE,
                                    "reasoning": f"[dreamer] {(best.reasoning or '')[:160]}"}
                        # Preserve validated vision coordinates through WebDreamer
                        # override. When WebDreamer's best action has no grounding
                        # (no element_id, no coords) but the PRIOR proposed action had
                        # validated vision coordinates, carry them forward. This prevents
                        # the "bare click" loop where WebDreamer correctly rejects an
                        # element but loses the coordinate grounding from Vision.
                        if (not best.element_id and best.x is None and best.y is None
                                and proposed.get("vision_coords")):
                            # x/y survive the {**proposed, ...} spread since the
                            # override dict doesn't set "x" or "y" — but we make
                            # this explicit for clarity and safety.
                            logger.info(
                                "🌙→👁️ WebDreamer inherited validated vision "
                                "coords (%.0f,%.0f)",
                                proposed.get("x", 0), proposed.get("y", 0),
                            )
                        webdreamer_update["webdreamer_overrides"] = state.get("webdreamer_overrides", 0) + 1
                        logger.info("🌙 WebDreamer OVERRIDE → %s", best.describe())
                    else:
                        logger.info("🌙 WebDreamer confirms the original (best_score=%.2f)",
                                    dr.best_score)
                except asyncio.TimeoutError:
                    logger.warning("🌙 WebDreamer timed out — keeping the original action")
    except Exception as e:  # noqa: BLE001 — simulation never breaks the step
        logger.debug("WebDreamer skipped (non-fatal): %s", e)

    # ── current Sub-Goal Lock — deterministic anti-amnesia backstop ──
    # Fires ONLY in the post-rejection danger zone (a 'done' was just rejected),
    # so it never interferes with normal multi-step work. If the worker proposes
    # RE-DOING an already-locked sub-goal, we don't execute it — hold (wait) and
    # redirect to the remaining work via the guidance bus.
    subgoal_lock_update: dict[str, Any] = {}
    try:
        from agent_first_browse.config.feature_flags import subgoal_lock_enabled
        if (subgoal_lock_enabled() and state.get("done_blocked", 0) > 0
                and state.get("prm_checklist")):
            from agent_first_browse.cognition.subgoal_lock import targets_locked_subgoal, remaining_subgoals
            hit = targets_locked_subgoal(proposed, state.get("prm_checklist"))
            if hit is not None:
                rem = remaining_subgoals(state.get("prm_checklist"))
                rem_str = "; ".join((d.get("desc") or "")[:60] for d in rem[:2]) or "the remaining work"
                logger.warning("🔒 Sub-Goal Lock: blocked a RE-DO of completed '%s' — "
                               "redirecting to remaining work.", (hit.get("desc") or "")[:50])
                proposed = {**proposed, "verb": "wait", "wait_ms": 600,
                            "risk_level": "REVERSIBLE", "reversible": True}
                subgoal_lock_update["correction_context"] = (
                    f"\n\n🔒 STOP — '{(hit.get('desc') or 'that sub-goal')[:60]}' is ALREADY "
                    f"DONE and LOCKED. Do NOT repeat it (its control may still be visible). "
                    f"Work ONLY on what remains: {rem_str}.")
    except Exception as e:  # noqa: BLE001 — backstop never breaks the step
        logger.debug("Sub-Goal Lock backstop skipped (non-fatal): %s", e)

    # Last writer wins in the pipeline (consensus/vision/dreamer may each replace
    # the worker action), so enforce the survey *execution* gate once more
    # immediately before returning the executable proposal. Profile learning is
    # deliberately not part of this safety gate: it is optional write-behind
    # metadata and must never convert a valid survey click into an endless wait.
    try:
        from agent_first_browse.cognition.stagnation import navigation_cycle_action_violation
        final_cycle_violation = navigation_cycle_action_violation(state, proposed)
        if final_cycle_violation:
            logger.warning(
                "🔂 FINAL NAVIGATION CYCLE GATE held blocked element [%s]",
                proposed.get("element_id") or "",
            )
            proposed = {
                **proposed,
                "verb": "wait",
                "element_id": None,
                "x": None,
                "y": None,
                "survey_gate_hold": True,
                "navigation_cycle_hold": True,
                "wait_ms": 500,
                "reasoning": final_cycle_violation[:200],
                "expected_change": "",
                "risk_level": "REVERSIBLE",
                "reversible": True,
            }
            webdreamer_update["correction_context"] = final_cycle_violation[:500]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Final navigation-cycle gate skipped (non-fatal): %s", exc)

    # Consensus, vision, and WebDreamer are all allowed to replace the worker's
    # proposal. Re-apply profile grounding after those last writers and before
    # the final execution gate.
    if str(proposed.get("verb") or "").lower() == "ask_user":
        proposed = {
            **proposed,
            "verb": "wait", "element_id": None, "x": None, "y": None,
            "wait_ms": 1000, "needs_vision": True,
            "vision_question": "Resolve the current interaction from current browser evidence.",
            "reasoning": "Autonomous recovery: re-perceive and choose a grounded browser action.",
            "expected_change": "Fresh perception reveals a grounded autonomous action or confirms a verified boundary.",
        }
        logger.warning("♾️ Removed late human-assistance override; continuing autonomously")

    validation_profile: dict[str, Any] = {}
    if survey_handoff and proposed.get("verb") in {
        "type", "click", "select_option", "set_date_of_birth",
    }:
        try:
            from agent_first_browse.survey.profile import (
                enforce_profile_choice,
                enforce_profile_date_action,
                enforce_typed_profile_fact,
                load_active_profile,
            )
            validation_profile = load_active_profile() or (
                state.get("survey_profile", {}) or {}
            )
            profile_note = ""
            profile_violation = ""
            if proposed.get("verb") == "type":
                proposed, profile_note, profile_violation = enforce_typed_profile_fact(
                    proposed, validation_profile, selector_map,
                    page_text=state.get("page_text", ""),
                )
            elif proposed.get("verb") in {"click", "select_option"}:
                proposed, profile_note, profile_violation = enforce_profile_choice(
                    proposed, validation_profile, selector_map,
                    page_text=state.get("page_text", ""),
                )
            elif proposed.get("verb") == "set_date_of_birth":
                proposed, profile_note, profile_violation = enforce_profile_date_action(
                    proposed, validation_profile
                )
            if profile_note:
                logger.info("🛡️ FINAL PROFILE GUARD: %s", profile_note)
            if profile_violation:
                logger.warning("🛡️ FINAL PROFILE GUARD held action: %s", profile_violation)
                proposed = {
                    **proposed,
                    "verb": "wait",
                    "element_id": None,
                    "x": None,
                    "y": None,
                    "survey_gate_hold": True,
                    "wait_ms": 500,
                    "reasoning": profile_violation[:200],
                    "expected_change": "",
                    "risk_level": "REVERSIBLE",
                    "reversible": True,
                }
                webdreamer_update["correction_context"] = (
                    "\n\n🛡️ AUTHORITATIVE PROFILE GATE: " + profile_violation
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Final typed profile guard skipped (non-fatal): %s", exc)

    if survey_handoff:
        try:
            from agent_first_browse.survey.context import survey_gate_violation
            final_violation = survey_gate_violation(
                proposed, selector_map,
                page_text=state.get("page_text", ""),
                audio_analysis=state.get("survey_audio_analysis") or {},
                continuous_mode=bool(state.get("continuous_survey_mode")),
                unavailable_offer_ids=survey_unavailable_offer_ids,
            )
            if captcha_page and proposed.get("verb") == "type" and not proposed.get("vision_verified"):
                final_violation = (
                    "Image-code verification was not successfully read by vision. "
                    "Do not guess or submit a placeholder; retry the visual read autonomously."
                )
            if captcha_page and proposed.get("verb") == "click":
                from agent_first_browse.survey.context import captcha_field_state, captcha_forward_id
                _captcha_input, _captcha_value = captcha_field_state(selector_map)
                if (
                    _captcha_value
                    and proposed.get("element_id") == captcha_forward_id(selector_map)
                    and not proposed.get("captcha_verified")
                ):
                    final_violation = (
                        "The CAPTCHA field is filled but has not passed an independent visual "
                        "character-for-character comparison. Verify it before submitting."
                    )
            if final_violation:
                logger.warning("🧷 FINAL SURVEY GATE held unsafe action: %s",
                               final_violation[:140])
                held_verb = str(proposed.get("verb") or "click")
                held_element_id = str(proposed.get("element_id") or decision.element_id or "")
                proposed = {
                    **proposed,
                    "verb": "wait",
                    "element_id": None,
                    "x": None,
                    "y": None,
                    "survey_gate_hold": True,
                    "wait_ms": 500,
                    "reasoning": final_violation[:200],
                    "expected_change": "",
                    "risk_level": "REVERSIBLE",
                    "reversible": True,
                }
                hold_identity = "|".join((
                    str(state.get("snapshot_revision") or state.get("survey_page_fingerprint") or ""),
                    held_verb,
                    held_element_id,
                ))
                prior_identity = str(state.get("survey_hold_identity") or "")
                prior_count = int(state.get("survey_hold_count", 0) or 0)
                hold_count = prior_count + 1 if hold_identity == prior_identity else 1
                proposed["held_action_identity"] = hold_identity
                proposed["hold_count"] = hold_count
                proposed["gate_reason_code"] = (
                    "SURVEY_NATIVE_CONTROLS_MISSING" if "native answer controls" in final_violation
                    else "SURVEY_GATE_REJECTED"
                )
                # Keep the original action identity in state even though the
                # executable proposal is safely converted to wait.
                proposed["survey_gate_hold"] = True
                webdreamer_update["correction_context"] = (
                    "\n\n🧷 SURVEY STATE GATE: " + final_violation
                    + " Re-read the current question and selected markers."
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Final survey state gate skipped (non-fatal): %s", e)

        try:
            from agent_first_browse.survey.profile import load_active_profile, sanitize_profile_update
            validation_profile = validation_profile or load_active_profile() or (
                state.get("survey_profile", {}) or {}
            )
            proposed, memory_note = sanitize_profile_update(
                proposed, validation_profile
            )
            if memory_note:
                logger.info(
                    "🧑‍💾 PROFILE MEMORY skipped; browser action proceeds: %s",
                    memory_note[:160],
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Profile metadata sanitization skipped (non-fatal): %s", e)

    # Consensus/vision/dreamer can replace the element after the initial target
    # metadata was built. Rebind its semantic context so outcome memory learns
    # the actual card/provider that was clicked, not the superseded candidate.
    final_element_id = proposed.get("element_id")
    if final_element_id and final_element_id in selector_map:
        final_element = selector_map[final_element_id]
        proposed["target_name"] = str(
            final_element.get("name", final_element.get("text", "")) or ""
        )[:60]
        proposed["target_context"] = str(
            final_element.get("hint") or final_element.get("container") or ""
        )[:120]

    # Validate model-proposed batching after every possible last-writer
    # (consensus/vision/dreamer). If the primary action changed, its old queue is
    # stale and is discarded. The validator may add one safe automatic Next.
    if survey_handoff:
        try:
            from agent_first_browse.survey.context import prepare_survey_transaction
            current_anchor = (
                proposed.get("verb"), proposed.get("element_id"), proposed.get("text")
            )
            auxiliary_reconsidered = bool(
                vision_attempted
                or consensus_update.get("consensus_votes")
                or webdreamer_update.get("webdreamer_runs")
            )
            raw_queue = (
                proposed.get("queued_actions", [])
                if current_anchor == queue_anchor and not auxiliary_reconsidered
                else []
            )
            if auxiliary_reconsidered:
                prepared_queue, queue_reason = [], "auxiliary model reconsidered the primary action"
            else:
                prepared_queue, queue_reason = prepare_survey_transaction(
                    proposed,
                    raw_queue,
                    selector_map,
                    page_text=state.get("page_text", ""),
                    continuous_mode=bool(state.get("continuous_survey_mode")),
                )
            proposed["queued_actions"] = prepared_queue
            proposed["execution_mode"] = "page_transaction" if prepared_queue else "single_action"
            if prepared_queue:
                logger.info(
                    "⚡ PAGE TRANSACTION prepared: primary + %d guarded follow-up(s)",
                    len(prepared_queue),
                )
            elif raw_queue and queue_reason:
                logger.info("⚡ PAGE TRANSACTION declined: %s", queue_reason[:140])
        except Exception as exc:
            proposed["queued_actions"] = []
            proposed["execution_mode"] = "single_action"
            logger.debug("Survey transaction preparation skipped (non-fatal): %s", exc)

    # ── Action dedup tracking ──
    logger.info("SURVEY_ACTION_EVENT %s", json.dumps({
        "event": "survey_action_proposal",
        "run_id": state.get("run_id", ""),
        "survey_attempt_id": state.get("survey_attempt_id", ""),
        "step_id": state.get("step_number", 0),
        "snapshot_revision": state.get("snapshot_revision", ""),
        "page_fingerprint": state.get("survey_page_fingerprint", ""),
        "control_count": state.get("element_count", 0),
        "proposal_source": proposed.get("proposal_source", "model"),
        "proposal_action": proposed.get("verb", ""),
        "proposal_element_id": proposed.get("element_id", ""),
        "gate_reason": proposed.get("gate_reason_code", ""),
        "gate_verdict": "hold" if proposed.get("survey_gate_hold") else "allow",
        "vision_requested": bool(proposed.get("vision_requested") or proposed.get("vision_used")),
        "hold_count": proposed.get("hold_count", state.get("survey_hold_count", 0)),
    }, separators=(",", ":")))
    current_sig = f"{proposed['verb']}:{(proposed.get('text') or 'none')[:30]}"
    last_sig = state.get("last_action_signature", "")
    if current_sig == last_sig:
        new_consecutive = state.get("consecutive_identical_actions", 0) + 1
    else:
        new_consecutive = 0

    return {
        "proposed_action": proposed,
        "last_action_signature": current_sig,
        "consecutive_identical_actions": new_consecutive,
        "survey_model_wait_seconds": model_wait_seconds,
        "survey_hold_identity": str(proposed.get("held_action_identity") or state.get("survey_hold_identity") or ""),
        "survey_hold_count": int(proposed.get("hold_count") or state.get("survey_hold_count", 0) or 0),
        "survey_gate_exhausted": bool(
            int(proposed.get("hold_count", 0) or 0) >= 2
            and proposed.get("survey_gate_hold")
        ),
        # expose the bound target so Overwatch (reality note) and the
        # done-judge can keep the agent locked on the right item.
        "bound_target": (bound_target.phrase if bound_target else ""),
        **consensus_update,
        **vision_update,
        **webdreamer_update,
        **subgoal_lock_update,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Specialist System Prompts
# ═══════════════════════════════════════════════════════════════════════════════
