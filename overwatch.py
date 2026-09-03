"""Overwatch — Multi-Layered Verification Subgraph for True Brain v16.0.

The Overwatch is the ONLY node that can commit state. Every worker
proposes an action; Overwatch validates it before it becomes permanent.

Verification layers (ordered cheap → expensive):
  Layer 1: Deterministic state validation (~0ms)
  Layer 2: Grounding validation (~5ms)
  Layer 3: Action execution + DOM ground-truth (CriticV12) (~200ms)
  Layer 4: CoVe action-trail gate (for 'done' actions) (~0ms)
  Layer 5: Loop detection + circuit breaker (~0ms)

Design:
  - Layers are ordered by cost: 80% of failures caught at zero LLM cost
  - Each layer can short-circuit with a verdict (retry/rollback/escalate)
  - Only Layer 3 actually executes the action on the live browser
  - The Overwatch NEVER calls the LLM for decisions — it uses
    deterministic checks + existing CriticV12 for post-action analysis

References:
  - Universal Verifier (arXiv 2604.06240): layered verification cuts
    false positives from 30%+ to 1–8%
  - SAGE-32B (arXiv 2601.04237): verification sensitivity formula
    ε' = ε(1−α) + ε·α·ε_retry
  - Six Sigma Agent (arXiv 2601.22290): consensus voting for irreversible actions
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any


# Actions that can be judged by whether the rendered page made the predicted
# transition. Keyboard actions are included deliberately: the latest run got
# stuck pressing Enter because only click/type contributed to the escalation
# streak.
_PROGRESS_ACTIONS = {
    "click", "type", "select_option", "press_enter", "press_key",
    "press_combo", "drag_and_drop", "upload_file", "set_date_of_birth",
}

try:
    from app.logger import get_logger
    logger = get_logger("overwatch")
except ImportError:
    logger = logging.getLogger("overwatch")


# ═══════════════════════════════════════════════════════════════════════════════
#  V18 Cognition — escalation ladder driver
# ═══════════════════════════════════════════════════════════════════════════════

def _escalate(state: dict, reason: str, updates: dict) -> dict:
    """Advance the deterministic escalation ladder for a stuck state.

    Writes the next DISTINCT tactic's directive into `correction_context`
    (reusing the existing worker-prompt injection slot), advances the ladder,
    and decays strategy confidence. When the ladder is exhausted it sets the
    verdict to 'rollback' so the rollback node re-strategizes. Returns the
    ladder dict so the caller can see whether a re-strategy was requested.
    """
    from cognition import advance_ladder, update_confidence, obstacle_key

    plan_steps = state.get("plan_steps", [])
    active = next((s for s in plan_steps
                   if s.get("status") in ("active", "in_progress")), None)
    plan_step_desc = active.get("desc", "") if active else ""
    obk = obstacle_key(state.get("current_url", ""), plan_step_desc)

    lad = advance_ladder(
        state.get("current_obstacle", ""), obk,
        state.get("ladder_rung", 0), state.get("tried_tactics", []),
    )
    updates["current_obstacle"] = lad["obstacle"]
    updates["ladder_rung"] = lad["rung"]
    updates["tried_tactics"] = lad["tried"]
    updates["strategy_confidence"] = update_confidence(
        state.get("strategy_confidence", 1.0), progress=False)

    if lad["restrategize"]:
        updates["correction_context"] = (
            f"\n\n⚠️ {reason} Prior tactics did not work — forming a NEW strategy."
        )
        updates["overwatch_verdict"] = "rollback"
    else:
        updates["correction_context"] = (
            f"\n\n🧗 ADAPTIVE TACTIC [{lad['tactic']}]: {lad['directive']}"
        )
        # V21: the 'vision' rung now triggers a REAL screenshot consult on the
        # next worker step (was a no-op text directive before).
        if lad["tactic"] == "vision":
            updates["force_vision"] = True
    logger.info("🧗 Escalation [%s] rung→%d confidence=%.2f (%s)",
                lad["tactic"], lad["rung"], updates["strategy_confidence"], reason)
    return lad


# ═══════════════════════════════════════════════════════════════════════════════
#  Overwatch Node — The verification gate
# ═══════════════════════════════════════════════════════════════════════════════

def _action_loop_signature(state: dict[str, Any], verb: str, element_id: Any) -> str:
    """Identify repeats without conflating the same control on new SPA questions."""
    try:
        from survey_context import survey_semantic_page_identity
        page_identity = survey_semantic_page_identity(
            state.get("current_url", ""),
            state.get("survey_page_fingerprint", ""),
            state.get("survey_interaction_fingerprint", ""),
        )
    except Exception:
        page_identity = state.get("survey_page_fingerprint", "")
    return f"{verb}|{element_id or ''}|{page_identity}"


def _note_survey_no_effect(
    state: dict[str, Any], proposed: dict[str, Any], updates: dict[str, Any]
) -> None:
    if not state.get("continuous_survey_mode"):
        return
    try:
        from survey_context import survey_action_attempt_key
        key = survey_action_attempt_key(state, proposed)
        if not key:
            return
        counts = dict(state.get("survey_action_no_effect_counts") or {})
        counts[key] = int(counts.get(key, 0) or 0) + 1
        while len(counts) > 40:
            counts.pop(next(iter(counts)))
        updates["survey_action_no_effect_counts"] = counts
    except Exception:
        return


def _record_survey_recipe_failure(
    state: dict[str, Any], proposed: dict[str, Any]
) -> None:
    """Teach recipe memory that a forward/replayed action did not advance."""
    if not state.get("continuous_survey_mode"):
        return
    target = str(proposed.get("target_name") or "").lower()
    is_forward = any(term in target for term in ("next", "continue", "submit", "finish", "complete"))
    if not (is_forward or proposed.get("recipe_signature")):
        return
    try:
        from survey_recipe_memory import get_survey_recipe_memory
        get_survey_recipe_memory().observe_failure(
            url=str(state.get("current_url") or ""),
            page_text=str(state.get("page_text") or ""),
            selector_map=state.get("selector_map", {}) or {},
            action=proposed,
        )
    except Exception as exc:
        logger.debug("Survey recipe failure observation skipped (non-fatal): %s", exc)


async def overwatch_node(state: dict[str, Any], page, critic, action_verifier=None) -> dict:
    """Multi-layered verification before committing state.

    Args:
        state: Current BrainState dict
        page: Live Playwright page reference
        critic: CriticV12 instance
        action_verifier: ActionVerifier instance (optional)

    Returns:
        Dict of state updates including overwatch_verdict
    """
    proposed = state.get("proposed_action")
    if not proposed:
        # A model timeout is not a verified browser step. Treating this as pass
        # sent the graph through commit_node, consumed the action budget, reset
        # recovery context, and produced long runs with dozens of fake steps.
        logger.warning("Overwatch: no proposed action — retrying without committing a step")
        return {
            "overwatch_verdict": "retry",
            "correction_context": (
                "The previous model attempt returned no executable action. Re-read the "
                "current page and choose one grounded supported action; request vision "
                "if the obstacle cannot be resolved from the element map."
            ),
        }

    try:
        from survey_context import survey_ineffective_action_violation
        ineffective_violation = survey_ineffective_action_violation(state, proposed)
        if ineffective_violation:
            logger.warning("🚫 Ineffective survey action quarantined: %s", ineffective_violation)
            return {
                "overwatch_verdict": "retry",
                "proposed_action": None,
                "action_outcome": "SEMANTIC_NO_EFFECT_QUARANTINE",
                "correction_context": ineffective_violation[:500],
            }
    except Exception as exc:
        logger.debug("Semantic action quarantine skipped (non-fatal): %s", exc)

    # Outcome-memory backstop: a worker/failover/vision override may still pick
    # the exact element learned to close an A→B→A→B navigation loop. Refuse it
    # before any side effect; same-labelled sibling elements remain legal.
    try:
        from stagnation import navigation_cycle_action_violation
        cycle_violation = navigation_cycle_action_violation(state, proposed)
        if cycle_violation:
            logger.warning("🔂 Navigation-cycle action blocked before execution: %s",
                           cycle_violation[:160])
            return {
                "overwatch_verdict": "retry",
                "proposed_action": None,
                "action_outcome": "NAVIGATION_CYCLE_BLOCKED",
                "correction_context": cycle_violation[:500],
            }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Navigation-cycle backstop skipped (non-fatal): %s", exc)

    # A deterministic survey gate may replace an unsafe click with a short
    # wait while requesting fresh perception. Do not execute/journal that wait:
    # counting it as progress is what turns an unresolved gate into a tight
    # retry loop and trips the no-progress circuit breaker. Re-enter the worker
    # with the same page state instead.
    if proposed.get("survey_gate_hold"):
        logger.warning("🧷 Overwatch: survey-gate hold -> fresh perception (no progress committed)")
        return {
            "overwatch_verdict": "retry",
            "proposed_action": None,
            "action_outcome": "SURVEY_GATE_HOLD",
            "correction_context": (
                "\n\nThe previous action was held by the survey safety gate. "
                "Refresh perception and inspect the currently selected choice; "
                "do not repeat the held click or wait."
            ),
        }

    verb = proposed.get("verb", "")
    element_id = proposed.get("element_id")
    selector_map = state.get("selector_map", {})
    step_number = state.get("step_number", 0)

    # A confirmed selection is a completed state transition. Clicking the same
    # choice again can toggle custom survey controls or produce no observable
    # change, which used to feed the retry/vision loop. Let the worker re-read
    # the page and choose the forward control instead.
    if verb == "click" and element_id:
        target_meta = selector_map.get(element_id) or {}
        if target_meta.get("selected") or target_meta.get("checked"):
            logger.warning("🚫 Duplicate selected-control click blocked: [%s]", element_id)
            return {
                "overwatch_verdict": "retry",
                "proposed_action": None,
                "action_outcome": "DUPLICATE_SELECTED_CONTROL_BLOCKED",
                "correction_context": (
                    f"Control {element_id} is already selected. Do not click it again; "
                    "re-read the live DOM and use the enabled forward control."
                ),
            }

    updates: dict[str, Any] = {}

    # Defense in depth: Overwatch is the last authority before Chrome. Pin
    # typed facts, factual choices, and DOB widgets to the active profile after
    # every possible model-side override.
    if verb in {"type", "click", "select_option", "set_date_of_birth"}:
        try:
            from survey_context import is_survey_mission
            if is_survey_mission(state.get("objective", "")):
                from survey_profile import (
                    enforce_profile_choice,
                    enforce_profile_date_action,
                    enforce_typed_profile_fact,
                    load_active_profile,
                )
                active_profile = load_active_profile() or (state.get("survey_profile", {}) or {})
                # A native SELECT is sometimes exposed as kind=input. Typing
                # into it can alter its displayed value without firing the
                # change event used by form validation. Convert that proposal
                # to the real select primitive before profile enforcement.
                target = selector_map.get(str(element_id or "")) or {}
                if verb == "type" and (
                    str(target.get("tag") or "").lower() == "select"
                    or str(target.get("control_type") or "").lower()
                    in {"select", "select-one"}
                ):
                    proposed = {**proposed, "verb": "select_option"}
                    verb = "select_option"
                    updates["proposed_action"] = proposed
                if verb == "type":
                    proposed, profile_note, profile_violation = enforce_typed_profile_fact(
                        proposed, active_profile, selector_map,
                        page_text=state.get("page_text", ""),
                    )
                elif verb in {"click", "select_option"}:
                    proposed, profile_note, profile_violation = enforce_profile_choice(
                        proposed, active_profile, selector_map,
                        page_text=state.get("page_text", ""),
                    )
                else:
                    proposed, profile_note, profile_violation = enforce_profile_date_action(
                        proposed, active_profile
                    )
                if profile_note:
                    updates["proposed_action"] = proposed
                    logger.info("🛡️ Overwatch profile guard: %s", profile_note)
                if profile_violation:
                    logger.warning("🛡️ Overwatch blocked ungrounded profile action: %s",
                                   profile_violation)
                    return {
                        "overwatch_verdict": "retry",
                        "proposed_action": None,
                        "action_outcome": "PROFILE_FACT_REQUIRED",
                        "correction_context": profile_violation[:500],
                    }
                verb = str(proposed.get("verb") or verb)
                element_id = proposed.get("element_id")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Overwatch profile guard skipped (non-fatal): %s", exc)

    # Known provider defects are corrected deterministically immediately before
    # journaling/execution. This is intentionally later than all model/failover
    # decisions so no downstream model can put the invalid value back.
    applied_site_quirk = ""
    try:
        from survey_site_quirks import apply_site_quirks_to_action
        proposed, applied_site_quirk = apply_site_quirks_to_action(
            proposed,
            url=state.get("current_url", ""),
            selector_map=selector_map,
            page_text=state.get("page_text", ""),
        )
        if applied_site_quirk:
            updates["proposed_action"] = proposed
            logger.info(
                "🧩 Applied URL-scoped survey quirk %s (typed value redacted)",
                applied_site_quirk,
            )
    except Exception as exc:  # noqa: BLE001 - optional workaround, never break execution
        logger.debug("Site quirk action transform skipped (non-fatal): %s", exc)

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 1: Deterministic State Validation (~0ms)
    # ═══════════════════════════════════════════════════════════════════

    page_fsm = state.get("page_fsm", "READY")
    if page_fsm == "ERROR":
        logger.warning("⚠️ Overwatch L1: page FSM in ERROR state")
        updates["overwatch_verdict"] = "rollback"
        updates["error_count"] = state.get("error_count", 0) + 1
        return updates

    if verb == "done":
        # Done actions skip to Layer 4 (CoVe check)
        return await _layer_4_cove_check(state, page, updates)

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 2: Grounding Validation (~5ms)
    # ═══════════════════════════════════════════════════════════════════

    if verb in ("click", "type"):
        if (
            verb == "click"
            and not element_id
            and proposed.get("vision_coords")
            and proposed.get("coord_validated")
            and proposed.get("x") is not None
            and proposed.get("y") is not None
        ):
            # A second, independent vision call has already verified this
            # non-DOM target. Requiring a nearby accessibility node here would
            # reject the very class of modal/canvas controls vision resolves.
            ground_result = {
                "grounded": True,
                "x": float(proposed["x"]),
                "y": float(proposed["y"]),
                "reason": "independently vision-validated coordinates",
            }
        else:
            from mcp_tools import mcp_ground_action
            ground_result = await mcp_ground_action(
                element_id=element_id,
                x=proposed.get("x"),
                y=proposed.get("y"),
                selector_map=selector_map,
                elements_list=state.get("elements_list", []),
            )

        if not ground_result["grounded"]:
            logger.warning(
                "🎯 Overwatch L2: grounding REJECT — %s",
                ground_result["reason"]
            )
            updates["overwatch_verdict"] = "retry"
            updates["error_count"] = state.get("error_count", 0) + 1
            updates["grounding_rejects"] = state.get("grounding_rejects", 0) + 1
            updates["failures"] = state.get("failures", []) + [{
                "action": f"{verb}({element_id or ''})",
                "why": ground_result["reason"],
                "count": 1,
            }]
            return updates

        # Update coordinates from grounding
        proposed["x"] = ground_result["x"]
        proposed["y"] = ground_result["y"]
        updates["proposed_action"] = proposed

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 3: Execute + DOM Ground-Truth Check (CriticV12)
    # ═══════════════════════════════════════════════════════════════════

    # Pre-action snapshot — V19: REUSE the perceive snapshot already in state
    # instead of re-extracting. This (a) saves one full DOM extraction per step,
    # and (b) crucially keeps window.__aid (the element-handle registry) exactly
    # as the LLM saw it, so the action resolves the SAME node the LLM chose
    # (a re-extract here would rebuild the registry right before the click).
    try:
        from cognitive_core import dom_data_to_a11y_format
        pre_dom = {
            "elements": state.get("elements_list", []),
            "element_count": state.get("element_count", 0),
            "markdown": state.get("dom_markdown", ""),
        }
        pre_a11y = dom_data_to_a11y_format(pre_dom)
        await critic.snapshot_before(pre_a11y)
    except Exception as snap_err:
        logger.warning("Overwatch L3: pre-snapshot failed: %s", snap_err)
        try:
            await critic.snapshot_before(None)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════
    #  V29 Atomic Intent Journal — WRITE-AHEAD record BEFORE the side-effect.
    #  If the action times out / crashes mid-execution, this record survives so the
    #  NEXT decision knows the action was attempted (and may have partially applied)
    #  and must NOT blindly repeat it. The in-state write rides the node's atomic
    #  super-step commit; the durable file is written immediately (pre-execution).
    # ═══════════════════════════════════════════════════════════════════
    intent_entry = None
    try:
        from feature_flags import intent_journal_enabled
        if intent_journal_enabled():
            from intent_journal import make_intent, should_journal, persist_intent
            if should_journal(verb):
                el_kind = ""
                if element_id and isinstance(selector_map, dict):
                    el_kind = (selector_map.get(element_id, {}) or {}).get("kind", "") or ""
                intent_entry = make_intent(proposed, step_number, element_kind=el_kind)
                updates["last_attempted_action"] = intent_entry  # survives node return
                persist_intent(intent_entry)                      # durable, atomic, pre-exec
                logger.info("📝 Intent journaled (pre-exec): %s [%s] hazard=%s",
                            intent_entry["verb"], intent_entry.get("element_id") or "",
                            intent_entry.get("hazard"))
    except Exception as e:  # noqa: BLE001 — journaling never blocks the action
        logger.debug("Intent journal write-ahead skipped (non-fatal): %s", e)

    # Execute the action via MCP tools
    execution_started_at = time.monotonic()
    action_outcome = await _execute_action(proposed, page)
    if applied_site_quirk:
        action_outcome += f"; SITE QUIRK APPLIED ({applied_site_quirk})"

    # A successful click may launch the real destination in a popup while the
    # opener changes its own DOM. Adopt that page before verification so Reality,
    # the Critic, and the next perception inspect the destination rather than
    # mistakenly continuing on the qualification/dashboard tab.
    try:
        import mcp_tools
        handoff = await mcp_tools.adopt_new_page_if_opened(page)
        if handoff.get("blocked_url"):
            action_outcome += f"; BLACKLISTED SURVEY CLOSED ({handoff['blocked_url'][:80]})"
        if handoff.get("switched") and handoff.get("page") is not None:
            page = handoff["page"]
            try:
                critic._page = page
            except Exception:
                pass
            destination = handoff.get("new_url", "") or "about:blank"
            action_outcome += f"; NEW TAB ADOPTED (navigated to {destination[:80]})"
    except Exception as exc:
        logger.debug("Post-action tab check skipped (non-fatal): %s", exc)

    queued_executed = 0
    if _action_execution_confirmed(action_outcome) and proposed.get("queued_actions"):
        queue_outcomes, queued_executed = await _execute_guarded_survey_queue(
            proposed, state, page
        )
        if queue_outcomes:
            action_outcome += "; " + "; ".join(queue_outcomes)
        if queued_executed:
            proposed["transaction_executed"] = queued_executed
            proposed["expected_change"] = (
                "The guarded page-local answer transaction is applied in order and, "
                "when its final forward control is reached, a new question or validation appears."
            )
            updates["proposed_action"] = proposed
    if proposed.get("recipe_signature") and not _action_execution_confirmed(action_outcome):
        try:
            from survey_recipe_memory import get_survey_recipe_memory
            get_survey_recipe_memory().record_replay_failure(
                str(proposed.get("recipe_signature") or ""),
                str(proposed.get("recipe_action_key") or ""),
            )
        except Exception as exc:
            logger.debug("Survey recipe failure update skipped (non-fatal): %s", exc)

    updates["action_outcome"] = action_outcome
    updates["page_fsm"] = "ACTION_PENDING"
    updates["total_actions"] = state.get("total_actions", 0) + 1 + queued_executed

    # Stamp the journal with the post-execution status so every downstream return
    # path (contradiction / ineffective / retry) carries an accurate ledger. The
    # verified-success path below clears it entirely.
    if intent_entry is not None:
        try:
            from intent_journal import classify_status
            intent_entry = {**intent_entry, "status": classify_status(action_outcome)}
            updates["last_attempted_action"] = intent_entry
        except Exception:  # noqa: BLE001
            pass

    # ═══════════════════════════════════════════════════════════════════
    #  V29 Smart-Scroll guard — stop scrolling into a wall (Mandate 3)
    #  A scroll that doesn't move the page, or that has reached the bottom, is not
    #  going to reveal the target. Track the streak and escalate a DIFFERENT tactic
    #  instead of looping the scroll. Flag-gated; off ⇒ identical to V28.
    # ═══════════════════════════════════════════════════════════════════
    if verb == "scroll":
        try:
            from feature_flags import smart_scroll_enabled
            if smart_scroll_enabled():
                ol = action_outcome.lower()
                unproductive = ("no-op" in ol or "reached page bottom" in ol
                                or "did not move" in ol)
                if unproductive:
                    streak = state.get("scroll_stuck_streak", 0) + 1
                    updates["scroll_stuck_streak"] = streak
                    if streak >= 2:
                        logger.warning("🧭 Smart-scroll: %d unproductive scrolls — the "
                                       "target isn't reachable by scrolling here; "
                                       "escalating a different tactic.", streak)
                        _escalate(state, "Scrolling is not revealing the target "
                                  "(page at bottom / won't move).", updates)
                        if updates.get("overwatch_verdict") != "rollback":
                            updates["overwatch_verdict"] = "retry"
                        updates["page_fsm"] = "READY"
                        return updates
                else:
                    updates["scroll_stuck_streak"] = 0
        except Exception as e:  # noqa: BLE001
            logger.debug("Smart-scroll guard skipped (non-fatal): %s", e)

    # Post-action DOM check
    post_markdown = ""
    post_page_text = ""
    post_selector_map: dict[str, dict] = {}
    post_url = state.get("current_url", "")
    try:
        import dom_parser
        from cognitive_core import dom_data_to_a11y_format
        post_dom = await dom_parser.extract(page, timeout=3.0)
        post_a11y = dom_data_to_a11y_format(post_dom)
        # V29 Reality Monitor inputs (free — we already extracted the post DOM).
        post_markdown = post_dom.get("markdown", "") or ""
        post_page_text = post_dom.get("page_text", "") or ""
        post_selector_map = {
            str(item.get("id") or item.get("ref") or ""): item
            for item in (post_dom.get("elements") or [])
            if item.get("id") or item.get("ref")
        }
        try:
            post_url = page.url or post_url
        except Exception:
            pass
    except Exception:
        post_a11y = None

    px = proposed.get("x")
    py = proposed.get("y")
    x_val = int(px) if px is not None else 0
    y_val = int(py) if py is not None else 0
    target_ref = element_id or f"({x_val},{y_val})"
    execution_confirmed = _action_execution_confirmed(action_outcome)
    verdict = await critic.evaluate(
        action=verb,
        target_ref=target_ref,
        target_name=proposed.get("text", "")[:30] if proposed.get("text") else proposed.get("target_name", ""),
        a11y_data_after=post_a11y,
        # Executor truth outranks unrelated DOM churn. Previously every verb
        # except abandon_survey was reported as skill_success=True, so an
        # explicit CLICK/DRAG FAILED could become "progress" merely because a
        # dashboard carousel or modal changed its element count.
        skill_success=execution_confirmed,
        skill_error=("" if execution_confirmed else action_outcome),
    )
    # V29 Phase A: surface the unified state-change score (CriticV12 diffing).
    updates["state_change_score"] = float(getattr(verdict, "state_change_score", 0.0) or 0.0)

    # ═══════════════════════════════════════════════════════════════════
    #  V29 Reality Monitor — did the page do what the worker PREDICTED?
    #  CriticV12 above only answers "did *anything* change?". This compares the
    #  worker's pre-committed `expected_change` against the LIVE screen. A
    #  CONTRADICTED screen (error / rejection / wrong redirect / out-of-stock)
    #  must NOT be committed as progress just because the DOM mutated — that is
    #  exactly the "blind execution" failure. We block it, note the discrepancy,
    #  and feed it back to the worker (via the guidance bus) to re-evaluate.
    #  Fully additive + flag-gated (V29_REALITY) — off ⇒ identical to V28.
    # ═══════════════════════════════════════════════════════════════════
    try:
        from feature_flags import reality_enabled, reality_llm_enabled
        if reality_enabled() and verb in ("click", "type", "press_enter"):
            from reality import (classify_reality, reconcile_with_llm,
                                 CONTRADICTED, CONFIRMED, UNCLEAR)
            rv = classify_reality(
                expected_change=proposed.get("expected_change", "") or "",
                verb=verb,
                action_outcome=action_outcome,
                pre_text=state.get("dom_markdown", "") or "",
                post_text=post_markdown,
                pre_url=state.get("current_url", ""),
                post_url=post_url,
                critic_success=verdict.success,
            )
            # Deterministic-first: spend a cheap LLM call ONLY on ambiguous deltas.
            if (rv.status == UNCLEAR and reality_llm_enabled()
                    and _JUDGE_CHAIN and _JUDGE_INVOKE_FN):
                rv = await reconcile_with_llm(
                    _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH,
                    objective=state.get("objective", ""),
                    expected_change=proposed.get("expected_change", "") or "",
                    action_outcome=action_outcome,
                    post_text=post_markdown,
                    fallback=rv,
                )
            updates["reality_status"] = rv.status

            if rv.status == CONTRADICTED:
                logger.warning("🚨 Overwatch REALITY: screen contradicts the "
                               "prediction — %s", rv.note[:160])
                # Temporal self-questioning + target lock: a contradiction on the
                # bound target must NOT cause a drift to a neighboring look-alike.
                note = rv.note[:300]
                bt = (state.get("bound_target") or "").strip()
                if bt:
                    note = (note + f" Your bound target is: {bt[:90]}. Do NOT now act "
                            "on a similar control for a DIFFERENT or neighboring item "
                            "— that abandons the goal. Re-examine YOUR target, or halt "
                            "and let the ensemble vote.")
                updates["reality_note"] = note[:500]
                updates["critic_no_progress"] = state.get("critic_no_progress", 0) + 1
                if verb in _PROGRESS_ACTIONS:
                    updates["ineffective_streak"] = state.get("ineffective_streak", 0) + 1
                updates["reflexion_triggers"] = state.get("reflexion_triggers", 0) + 1
                _note_survey_no_effect(state, proposed, updates)
                _record_survey_recipe_failure(state, proposed)
                # Escalate a DISTINCT tactic and re-evaluate the real screen.
                _escalate(state,
                          "The screen did NOT do what the action predicted.",
                          updates)
                if updates.get("overwatch_verdict") != "rollback":
                    updates["overwatch_verdict"] = "retry"
                return updates

            if rv.status == CONFIRMED:
                # Positive reconciliation — clear any stale discrepancy note.
                updates["reality_note"] = ""
                # V32: Reality confirmed the action worked → clear the intent
                # journal so HESITATION doesn't fire on the next step. Without
                # this, the Worker gets told "predecessor click unconfirmed"
                # even though Overwatch visually confirmed the click succeeded.
                # This was the root cause of the "false hesitation" loop on
                # radio buttons and checkboxes.
                updates["last_attempted_action"] = None
                try:
                    from intent_journal import resolve_intent
                    resolve_intent()
                except Exception:  # noqa: BLE001
                    pass
                if rv.note:
                    logger.info("✅ Overwatch REALITY: prediction confirmed — %s",
                                rv.note[:120])
    except Exception as e:  # noqa: BLE001 — reality monitor never breaks the step
        logger.debug("Reality monitor skipped (non-fatal): %s", e)

    if verdict.success:
        updates["correction_failures"] = 0
        updates["correction_context"] = ""
        updates["critic_progress"] = state.get("critic_progress", 0) + 1
        # A generic DOM mutation is not enough to clear this trigger. Keyboard
        # actions can return OK while leaving the same screen in place. Only a
        # reality confirmation proves meaningful progress for these actions.
        if (updates.get("reality_status") == "CONFIRMED"
                or verb not in _PROGRESS_ACTIONS):
            updates["ineffective_streak"] = 0
        logger.info("✓ Overwatch L3: %s [%.0f%%]", verdict.reason[:100], verdict.confidence * 100)
    else:
        _note_survey_no_effect(state, proposed, updates)
        _record_survey_recipe_failure(state, proposed)
        cf = state.get("correction_failures", 0) + 1
        updates["correction_failures"] = cf
        updates["critic_no_progress"] = state.get("critic_no_progress", 0) + 1
        # V21: a click/type that produced no visible effect — count it so that a
        # short streak escalates to a vision look (the DOM says act, page doesn't).
        if verb in _PROGRESS_ACTIONS:
            updates["ineffective_streak"] = state.get("ineffective_streak", 0) + 1

        if verdict.circuit_breaker_triggered:
            logger.warning("🛑 Overwatch L3: circuit breaker — %s", verdict.reason[:100])
            _escalate(state, "Critic circuit breaker.", updates)
            updates["overwatch_verdict"] = "rollback"
            updates["reflexion_triggers"] = state.get("reflexion_triggers", 0) + 1
            return updates

        if cf >= 2:
            # Escalate through the ladder (distinct tactic each time; never repeats).
            _escalate(state, f"Last {cf} actions had no visible effect.", updates)
            updates["reflexion_triggers"] = state.get("reflexion_triggers", 0) + 1
            # Honor a ladder-forced rollback; otherwise retry until cf exhausts.
            if updates.get("overwatch_verdict") != "rollback":
                updates["overwatch_verdict"] = "retry" if cf < 3 else "rollback"
            return updates

        # cf == 1: one free re-decision before we start escalating.
        updates["overwatch_verdict"] = "retry"
        return updates

    # ═══════════════════════════════════════════════════════════════════
    #  Layer 5: Loop Detection + Circuit Breaker (~0ms)
    # ═══════════════════════════════════════════════════════════════════

    # Element IDs are snapshot-local. Survey SPAs reuse e.g. e18='Next' across
    # many different questions under one URL, so include the question identity
    # to avoid treating legitimate forward progress as an action loop.
    sig = _action_loop_signature(state, verb, element_id)
    loop_sigs = state.get("loop_signatures", [])
    sig_count = loop_sigs[-12:].count(sig) if loop_sigs else 0  # sliding window

    if sig_count >= 3:
        logger.warning("🔄 Overwatch L5: loop detected — sig '%s' x%d", sig[:60], sig_count)
        _escalate(state, "Loop detected (same action repeating).", updates)
        updates["overwatch_verdict"] = "rollback"
        updates["loop_signatures"] = [sig]
        return updates

    # ── All layers passed ──
    updates["overwatch_verdict"] = "pass"
    updates["page_fsm"] = "VALIDATED"
    from brain_state import LOOP_SIGNATURE_MAX, append_bounded
    updates["loop_signatures"] = append_bounded(
        loop_sigs, [sig], LOOP_SIGNATURE_MAX
    )

    # V29: the action is VERIFIED-effective → resolve the intent journal. There is
    # no ambiguity to carry forward, so the next decision starts with a clean slate.
    if intent_entry is not None:
        updates["last_attempted_action"] = None
        try:
            from intent_journal import resolve_intent
            resolve_intent()
        except Exception:  # noqa: BLE001
            pass

    # A character answer becomes durable only here, after every execution and
    # reality layer has accepted the browser action. Failed, ambiguous, held, or
    # merely proposed answers can therefore never pollute the respondent profile.
    try:
        from survey_context import is_survey_mission
        if is_survey_mission(state.get("objective", "")):
            from survey_profile import (
                commit_confirmed_survey_answer,
                compact_runtime_profile,
                render_profile,
            )
            learned_profile, learned, learning_note = commit_confirmed_survey_answer(
                state.get("survey_profile", {}) or {}, proposed
            )
            if learned:
                current_question = str(proposed.get("question_text") or "")
                updates["survey_profile"] = compact_runtime_profile(
                    learned_profile, current_question
                )
                updates["survey_profile_render"] = render_profile(
                    learned_profile, current_question
                )
                logger.info("🧑‍💾 PROFILE MEMORY: %s", learning_note[:160])
            elif learning_note not in {
                "profile learning disabled",
                "answer does not establish character memory",
                "age already derives from date_of_birth",
            }:
                logger.warning("Profile memory not committed: %s", learning_note[:160])
    except Exception as e:  # noqa: BLE001 — profile persistence never breaks a step
        logger.warning("Profile memory commit failed (browser action remains valid): %s", e)

    # Record step in history
    pre_page_fingerprint = ""
    post_page_fingerprint = ""
    pre_semantic_identity = ""
    post_semantic_identity = ""
    survey_transition_verified = False
    survey_completion_verified = False
    try:
        from survey_context import (
            is_verified_survey_page_transition,
            survey_action_signature,
            survey_completion_evidence,
            survey_interaction_fingerprint,
            survey_page_fingerprint,
            survey_semantic_page_identity,
        )
        pre_page_fingerprint = survey_page_fingerprint(
            state.get("page_text", ""), state.get("selector_map", {}) or {}
        )
        post_page_fingerprint = survey_page_fingerprint(
            post_page_text or post_markdown, post_selector_map
        )
        pre_semantic_identity = survey_semantic_page_identity(
            state.get("current_url", ""),
            pre_page_fingerprint,
            survey_interaction_fingerprint(state.get("selector_map", {}) or {}),
        )
        post_semantic_identity = survey_semantic_page_identity(
            post_url,
            post_page_fingerprint,
            survey_interaction_fingerprint(post_selector_map),
        )
        survey_transition_verified = is_verified_survey_page_transition(
            pre_page_fingerprint,
            post_page_fingerprint,
            action_outcome,
            previous_url=str(state.get("current_url") or ""),
            current_url=post_url,
            current_page_text=post_page_text or post_markdown,
            action=proposed,
        )
        survey_completion_verified = bool(
            survey_completion_evidence(post_page_text or post_markdown)
        )
    except Exception:
        survey_action_signature = lambda _action: ""  # type: ignore[assignment]
    history_entry = {
        "step": step_number + 1,
        "verb": verb,
        "element_id": element_id,
        "target_name": proposed.get("target_name", "")[:100],
        "target_context": proposed.get("target_context", "")[:120],
        "action": f"{verb}" + (f"({x_val},{y_val})" if px is not None else ""),
        "outcome": action_outcome,
        "reality_status": updates.get("reality_status", ""),
        "screen": proposed.get("screen_state", "")[:80],
        "question_text": proposed.get("question_text", "")[:300],
        "answer_value": str(
            proposed.get("text")
            if verb in {"type", "select_option"} and proposed.get("text") is not None
            else proposed.get("target_name", "")
        )[:120],
        "answer_basis": proposed.get("answer_basis", "")[:80],
        "profile_update_category": proposed.get("profile_update_category", "none")[:40],
        "profile_update_key": proposed.get("profile_update_key", "")[:80],
        "profile_update_mode": proposed.get("profile_update_mode", "none")[:20],
        "pre_url": state.get("current_url", ""),
        "pre_page_fingerprint": pre_page_fingerprint,
        "post_page_fingerprint": post_page_fingerprint,
        "pre_semantic_identity": pre_semantic_identity,
        "post_semantic_identity": post_semantic_identity,
        "action_signature": survey_action_signature(proposed),
        "survey_transition_verified": survey_transition_verified,
        "survey_completion_verified": survey_completion_verified,
        "url": page.url if page else "",
    }
    from brain_state import HISTORY_MAX_ENTRIES, append_bounded
    updates["history"] = append_bounded(
        state.get("history", []), [history_entry], HISTORY_MAX_ENTRIES
    )
    if history_entry["question_text"] and history_entry["answer_value"]:
        from brain_state import SURVEY_CYCLE_ARCHIVE_MAX
        updates["survey_cycle_answers"] = append_bounded(
            state.get("survey_cycle_answers", []),
            [{
                "question_text": history_entry["question_text"],
                "answer_value": history_entry["answer_value"],
                "answer_basis": history_entry["answer_basis"],
            }],
            SURVEY_CYCLE_ARCHIVE_MAX,
        )

    if survey_transition_verified or survey_completion_verified:
        try:
            from survey_recipe_memory import get_survey_recipe_memory
            get_survey_recipe_memory().observe_success(
                url=state.get("current_url", ""),
                page_text=state.get("page_text", ""),
                selector_map=state.get("selector_map", {}) or {},
                action=proposed,
                elapsed_ms=(time.monotonic() - execution_started_at) * 1000.0,
                verified_transition=True,
            )
        except Exception as exc:
            logger.debug("Survey recipe observation skipped (non-fatal): %s", exc)
    else:
        _record_survey_recipe_failure(state, proposed)

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  Layer 4: Outcome verification (for 'done' actions)
# ═══════════════════════════════════════════════════════════════════════════════
#  V20: the gate is now an EVIDENCE-GROUNDED judge over the FRESH page state
#  (objective + planner success-criteria + live DOM + action trail), not the old
#  process-trail heuristics. The heuristics remain ONLY as a fallback when no
#  LLM verdict can be obtained. See outcome_judge.py for the full rationale.

# Judge dependencies — injected once at startup by brain_graph (same pattern as
# mcp_tools.set_page). Without configuration the gate falls back to heuristics.
_JUDGE_INVOKE_FN = None
_JUDGE_CHAIN: list = []
_JUDGE_BREAKER = None
_JUDGE_HEALTH = None


def configure_outcome_judge(invoke_fn, failover_chain, breaker, health_tracker) -> None:
    """Wire the Plan-1 model layer into the done-gate (called at brain startup)."""
    global _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH
    _JUDGE_INVOKE_FN = invoke_fn
    _JUDGE_CHAIN = failover_chain or []
    _JUDGE_BREAKER = breaker
    _JUDGE_HEALTH = health_tracker


def _block_done(state: dict, updates: dict, reason: str,
                correction: str = "") -> dict:
    """Reject a 'done', with a TERMINATION GUARANTEE: after MAX_DONE_BLOCKS
    rejections we stop looping and finalize honestly (mission_success=False).
    The old counter was incremented but never read — block-loops could spin
    forever (observed: done ×5 blocked on the HN run).

    V31 WORKER VETO: If the worker has provided substantive proof_of_completion
    and been rejected 2+ times, force-accept. The Worker has temporal context
    that the static Judge lacks (e.g. saw cart count change, button state flip).
    """
    from outcome_judge import MAX_DONE_BLOCKS
    blocked = state.get("done_blocked", 0) + 1
    updates["done_blocked"] = blocked
    from brain_state import HISTORY_MAX_ENTRIES, append_bounded
    updates["history"] = append_bounded(state.get("history", []), [{
        "step": state.get("step_number", 0) + 1,
        "action": "done",
        "outcome": f"BLOCKED: {reason[:80]}",
    }], HISTORY_MAX_ENTRIES)

    # ── V31 WORKER VETO: Confidence Override ──
    # If the worker has fired done 2+ times with substantive proof, the worker's
    # temporal context overrides the judge's static snapshot. This prevents
    # infinite stagnation on successfully completed tasks (e.g. Amazon smart-wagon
    # redirect stripping the product name from DOM).
    proof = (state.get("proposed_action") or {}).get("proof_of_completion", "").strip()
    if blocked >= 2 and len(proof) > 30:
        logger.warning(
            "🏆 Overwatch L4: WORKER VETO — done rejected %d× but worker provided "
            "substantive proof (%d chars). Accepting worker's temporal evidence. "
            "Proof: %s", blocked, len(proof), proof[:200],
        )
        updates["overwatch_verdict"] = "pass"
        updates["mission_success"] = True
        updates["done_evidence"] = f"WORKER VETO (proof accepted after {blocked} rejections): {proof[:300]}"
        return updates

    if blocked >= MAX_DONE_BLOCKS:
        logger.warning(
            "🛑 Overwatch L4: done blocked %d× — finalizing UNVERIFIED (honest failure). "
            "Last gap: %s", blocked, reason[:120],
        )
        updates["overwatch_verdict"] = "escalate"   # routes straight to finalize
        updates["mission_success"] = False
        updates["done_evidence"] = f"UNVERIFIED after {blocked} attempts: {reason[:160]}"
        return updates
    logger.warning("🛡️ Overwatch L4: done BLOCKED (%d/%d) — %s",
                   blocked, MAX_DONE_BLOCKS, reason[:120])
    updates["overwatch_verdict"] = "retry"
    if correction:
        updates["correction_context"] = correction
    return updates


def _render_verified_subgoals(state: dict) -> str:
    """Sub-goals already verified complete during execution, with their captured
    proof — trusted prior evidence handed to the done-judge so it confirms rather
    than re-deriving subtle UI changes from a cold final page. Generalized: read
    straight from the task's own checklist, no website/action rules."""
    # ONLY genuinely-locked sub-goals — never the background audit's unverified
    # "done" guesses (those would bias the judge toward false completion). The
    # done-judge otherwise stays purely evidence-grounded on the live page.
    lines = []
    for d in (state.get("prm_checklist", []) or []):
        if d.get("verified"):
            ev = (d.get("evidence") or "").strip()
            lines.append(f"  ✓ {(d.get('desc') or '').strip()}" + (f" — {ev}" if ev else ""))
    return "\n".join(lines)


async def final_outcome_audit(state: dict, page) -> tuple[bool, str]:
    """Last-chance outcome verification when the agent shuts down WITHOUT a
    verified 'done' (budget exhausted, restrategize cap, …).

    The agent failing to *declare* done does not mean the task failed — on
    sites where the confirmation click misfires, the goal evidence may already
    be on screen (e.g. Flipkart's button switched to 'Go to cart'). One judge
    call here converts those false-failures into verified successes, and makes
    every shutdown report honest: verified-success / verified-failure /
    unverifiable. Returns (achieved, evidence_or_gap)."""
    from outcome_judge import judge_done, should_accept

    page_text = ""
    dom_markdown = state.get("dom_markdown", "")
    url = state.get("current_url", "")
    try:
        url = page.url or url
        import dom_parser
        fresh = await dom_parser.extract(page, timeout=4.0)
        if fresh.get("elements"):
            dom_markdown = fresh.get("markdown", dom_markdown)
        page_text = await page.evaluate(
            "() => (document.body && document.body.innerText || '')"
            ".replace(/\\s+/g, ' ').slice(0, 4500)"
        )
    except Exception as e:
        logger.debug("final audit perception failed (%s) — using last state", e)

    verdict = await judge_done(
        _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH,
        objective=state.get("objective", ""),
        success_criteria=state.get("success_criteria", ""),
        url=url,
        dom_markdown=dom_markdown,
        history_tail=state.get("history_compressed", ""),
        claim="(none — the agent shut down without declaring done)",
        page_text=page_text,
        verified_subgoals=_render_verified_subgoals(state),
        bound_target=state.get("bound_target", ""),
    )
    if verdict is None:
        return False, "unverifiable (judge offline)"
    if should_accept(verdict):
        return True, verdict.evidence[:300]
    return False, (verdict.missing or verdict.evidence)[:300]


async def _layer_4_cove_check(state: dict, page, updates: dict) -> dict:
    """Outcome verification before accepting 'done'."""
    from outcome_judge import judge_done, should_accept, rejection_feedback

    objective = state.get("objective", "")

    if state.get("continuous_survey_mode"):
        correction = (
            "\n\n♾️ CONTINUOUS SURVEY MODE: 'done' is not a valid autonomous action. "
            "If a paid survey has just started, complete it. If one survey was credited, "
            "return to the dashboard and select the next best reward-per-minute offer."
        )
        updates["overwatch_verdict"] = "retry"
        updates["mission_success"] = False
        updates["correction_context"] = correction
        from brain_state import HISTORY_MAX_ENTRIES, append_bounded
        updates["history"] = append_bounded(state.get("history", []), [{
            "step": state.get("step_number", 0) + 1,
            "action": "done",
            "outcome": "BLOCKED: continuous survey cycle must continue",
        }], HISTORY_MAX_ENTRIES)
        logger.warning("♾️ Overwatch blocked terminal done during continuous survey mode")
        return updates

    # ── Cheap, high-precision pre-block: an auth wall is never success ──
    try:
        url = page.url if page else state.get("current_url", "")
    except Exception:
        url = state.get("current_url", "")
    if any(kw in (url or "").lower() for kw in ("login", "signin", "auth", "signup")):
        return _block_done(
            state, updates, f"Still on auth page ({url[:60]})",
            "\n\n🛡️ DONE REJECTED: you are on a login/auth page — the task cannot "
            "be complete here. Deal with the auth wall or navigate back to the task.",
        )

    # ── Fresh perception: judge the page AS IT IS NOW (a confirmation toast /
    #    cart badge may have appeared after the last snapshot) ──
    dom_markdown = state.get("dom_markdown", "")
    try:
        import dom_parser
        fresh = await dom_parser.extract(page, timeout=4.0)
        if fresh.get("elements"):
            dom_markdown = fresh.get("markdown", dom_markdown)
    except Exception as e:
        logger.debug("L4 fresh snapshot failed (%s) — using last snapshot", e)

    # Visible page text — proof routinely lives in NON-interactive content
    # (confirmation messages, a fact the task asked to find, cart line-items).
    # The element map alone made the judge blind to all of it.
    page_text = ""
    try:
        page_text = await page.evaluate(
            "() => (document.body && document.body.innerText || '')"
            ".replace(/\\s+/g, ' ').slice(0, 4500)"
        )
    except Exception as e:
        logger.debug("L4 page-text read failed (%s)", e)

    proposed = state.get("proposed_action") or {}
    claim = " | ".join(
        s for s in (
            (proposed.get("screen_state", "") or "").strip(),
            (proposed.get("goal_progress", "") or "").strip(),
            (proposed.get("reasoning", "") or "").strip(),
        ) if s
    )

    # V31: Extract worker's proof_of_completion for the judge
    proof_of_completion = (proposed.get("proof_of_completion", "") or "").strip()
    if proof_of_completion:
        logger.info("📋 Worker proof_of_completion: %s", proof_of_completion[:200])

    verdict = await judge_done(
        _JUDGE_INVOKE_FN, _JUDGE_CHAIN, _JUDGE_BREAKER, _JUDGE_HEALTH,
        objective=objective,
        success_criteria=state.get("success_criteria", ""),
        url=url,
        dom_markdown=dom_markdown,
        history_tail=state.get("history_compressed", ""),
        claim=claim,
        page_text=page_text,
        verified_subgoals=_render_verified_subgoals(state),
        bound_target=state.get("bound_target", ""),
        proof_of_completion=proof_of_completion,
    )

    if verdict is not None:
        if should_accept(verdict):
            logger.info("✅ Overwatch L4: outcome VERIFIED — %s", verdict.evidence[:140])
            updates["overwatch_verdict"] = "pass"
            updates["mission_success"] = True
            updates["done_evidence"] = verdict.evidence[:300]
            return updates
        # V29 Sub-Goal Lock: a global rejection must NOT erase verified sub-goals.
        # Re-affirm the locked-done work + name only what REMAINS (Partial Success),
        # and align the plan to the ledger — instead of a bare "missing X" that makes
        # the agent re-do an already-finished sub-goal (the amnesia loop).
        correction = rejection_feedback(verdict)
        try:
            from feature_flags import subgoal_lock_enabled
            if subgoal_lock_enabled() and state.get("prm_checklist"):
                from subgoal_lock import compose_rejection, reconcile_plan_with_ledger
                correction = compose_rejection(verdict.missing, verdict.next_hint,
                                               state["prm_checklist"])
                reconciled = reconcile_plan_with_ledger(state.get("plan_steps"),
                                                        state["prm_checklist"])
                if reconciled is not None:
                    updates["plan_steps"] = reconciled
                    logger.info("🔒 Sub-Goal Lock: plan realigned to the verified ledger "
                                "(locked sub-goals marked done).")
        except Exception as e:  # noqa: BLE001 — lock logic never breaks the gate
            logger.debug("Sub-Goal Lock rejection compose skipped: %s", e)

        return _block_done(
            state, updates,
            verdict.missing or "no on-page proof of completion",
            correction,
        )

    # ── Fallback (judge unavailable): legacy heuristic trail-check ──
    from execution_safety import cove_pre_done_check
    from cognitive_core import PlanState, WorkingMemory

    plan = PlanState(
        mission=objective,
        steps=state.get("plan_steps", []),
        current_step_id=state.get("plan_cursor", 0),
    )
    working_mem = WorkingMemory()
    for h in state.get("history", []):
        working_mem.episodic.append(h)

    cove_ok, cove_reason = await cove_pre_done_check(page, plan, working_mem, objective)

    if not cove_ok:
        return _block_done(state, updates, f"(heuristic) {cove_reason}")

    logger.info("✅ Overwatch L4: heuristic-verified done (judge offline) — %s",
                cove_reason[:60])
    updates["overwatch_verdict"] = "pass"
    updates["mission_success"] = True
    updates["done_evidence"] = f"(heuristic only) {cove_reason[:200]}"
    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Executor — delegates to MCP tools
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_fail(clean: bool, verb: str, legacy_prefix: str, error: str) -> str:
    """Failure outcome: clean semantic FailureClass when hybrid feedback is on, else
    the legacy string. Both preserve the raw error so downstream detectors work."""
    if not clean:
        return f"→ {legacy_prefix}: {error}"
    from action_feedback import classify_failure, render_failure
    return render_failure(classify_failure(verb, error), error)


def _action_execution_confirmed(outcome: str) -> bool:
    """Only an executor-confirmed action may be credited as progress."""
    return str(outcome or "").lstrip().startswith("→ OK")


async def _execute_guarded_survey_queue(
    proposed: dict[str, Any], state: dict[str, Any], page
) -> tuple[list[str], int]:
    """Execute a validated page-local queue, rechecking live state each time."""
    queued = list(proposed.get("queued_actions") or [])[:8]
    if not queued or not state.get("continuous_survey_mode"):
        return [], 0

    outcomes: list[str] = []
    executed = 0
    initial_page_fingerprint = str(state.get("survey_page_fingerprint") or "")
    for index, queued_action in enumerate(queued):
        try:
            from mcp_tools import mcp_snapshot
            from survey_context import (
                _element_label,
                blocking_popup_action_id,
                survey_gate_violation,
                survey_page_fingerprint,
            )

            snapshot = await mcp_snapshot()
            selector_map = snapshot.get("selector_map", {}) or {}
            page_text = snapshot.get("page_text", "") or ""
            live_fingerprint = survey_page_fingerprint(page_text, selector_map)
            if (
                initial_page_fingerprint
                and live_fingerprint
                and live_fingerprint != initial_page_fingerprint
            ):
                outcomes.append("TRANSACTION STOPPED (page changed before next queued action)")
                break
            if blocking_popup_action_id(selector_map):
                outcomes.append("TRANSACTION STOPPED (new blocking popup appeared)")
                break

            action = dict(queued_action or {})
            element_id = str(action.get("element_id") or "")
            expected_label = str(action.get("target_name") or "").strip().lower()
            live_element = selector_map.get(element_id) or {}
            if expected_label and _element_label(live_element) != expected_label:
                matches = [
                    str(eid) for eid, element in selector_map.items()
                    if _element_label(element) == expected_label
                ]
                if len(matches) != 1:
                    outcomes.append("TRANSACTION STOPPED (queued target became ambiguous)")
                    break
                element_id = matches[0]
                live_element = selector_map[element_id]
            if action.get("verb") in {"click", "type", "select_option"} and not live_element:
                outcomes.append("TRANSACTION STOPPED (queued target disappeared)")
                break
            action["element_id"] = element_id or None
            action["target_name"] = _element_label(live_element)[:160]

            if action.get("verb") == "type":
                try:
                    from survey_profile import enforce_typed_profile_fact, load_active_profile
                    action, profile_note, profile_violation = enforce_typed_profile_fact(
                        action,
                        load_active_profile() or (state.get("survey_profile", {}) or {}),
                        selector_map,
                        page_text=page_text,
                    )
                    if profile_violation or not profile_note:
                        outcomes.append("TRANSACTION STOPPED (profile fact required)")
                        break
                except Exception:
                    outcomes.append("TRANSACTION STOPPED (queued profile validation failed)")
                    break
            elif action.get("verb") in {"click", "select_option"}:
                try:
                    from survey_profile import enforce_profile_choice, load_active_profile
                    action, _note, profile_violation = enforce_profile_choice(
                        action,
                        load_active_profile() or (state.get("survey_profile", {}) or {}),
                        selector_map,
                        page_text=page_text,
                    )
                    if profile_violation:
                        outcomes.append("TRANSACTION STOPPED (profile choice conflict)")
                        break
                except Exception:
                    outcomes.append("TRANSACTION STOPPED (queued profile validation failed)")
                    break

            violation = survey_gate_violation(
                action,
                selector_map,
                page_text=page_text,
                audio_analysis=state.get("survey_audio_analysis") or {},
                continuous_mode=True,
            )
            if violation:
                outcomes.append(f"TRANSACTION STOPPED ({violation[:100]})")
                break

            try:
                from survey_site_quirks import apply_site_quirks_to_action
                action, _quirk = apply_site_quirks_to_action(
                    action,
                    url=getattr(page, "url", "") or state.get("current_url", ""),
                    selector_map=selector_map,
                    page_text=page_text,
                )
            except Exception:
                pass

            result = await _execute_action(action, page)
            if not _action_execution_confirmed(result):
                outcomes.append(f"TRANSACTION STOPPED ({result[:120]})")
                break
            executed += 1
            outcomes.append(f"TRANSACTION {index + 1}/{len(queued)} OK")
        except Exception as exc:
            outcomes.append(f"TRANSACTION STOPPED ({str(exc)[:100]})")
            break
    return outcomes, executed


async def _execute_action(proposed: dict, page) -> str:
    """Execute a proposed action via MCP tools. Returns an outcome string.

    V29 Phase A: asymmetric verbosity — terse on success (observable effect only,
    no internal strategy name), semantic FailureClass on failure — gated by
    `hybrid_primitives_enabled`. Plus expanded primitives (hover/select_option/
    press_key). Success keeps the "→ OK" prefix and failures keep the raw error so
    the win-state / Reality / Intent-Journal detectors are unaffected.
    """
    verb = proposed.get("verb", "wait")
    try:
        from feature_flags import hybrid_primitives_enabled
        clean = hybrid_primitives_enabled()
    except Exception:
        clean = False

    try:
        if verb == "goto":
            from mcp_tools import mcp_navigate
            result = await mcp_navigate(proposed.get("url", ""))
            if result["success"]:
                return f"→ OK (navigated to {result['url'][:60]})"
            return _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "abandon_survey":
            from mcp_tools import mcp_abandon_survey
            boundary_reason = str(proposed.get("survey_boundary_reason") or "")
            fresh_dashboard = boundary_reason == "completed:fresh_dashboard"
            try:
                from survey_site_quirks import fresh_dashboard_after_boundary
                fresh_dashboard = bool(
                    fresh_dashboard
                    or fresh_dashboard_after_boundary(
                        proposed.get("url", ""), boundary_reason
                    )
                )
            except Exception:
                pass
            result = await mcp_abandon_survey(
                proposed.get("url", ""),
                fresh_dashboard=fresh_dashboard,
            )
            if result["success"]:
                return f"→ OK (survey closed; provider restored at {result['url'][:60]})"
            return _fmt_fail(clean, verb, "ABANDON FAILED", result["error"])

        elif verb == "click":
            from mcp_tools import mcp_click
            answer_basis = str(proposed.get("answer_basis") or "").lower()
            target_name = str(proposed.get("target_name") or "").lower()
            answer_choice = bool(
                answer_basis
                and answer_basis not in {
                    "page_navigation", "reward_per_minute", "unknown_needs_vision",
                }
                and not any(
                    token in target_name
                    for token in ("next", "continue", "submit", "finish")
                )
            )
            replay_safe_navigation = bool(
                answer_basis == "page_navigation"
                and not any(
                    token in target_name
                    for token in (
                        "next", "continue", "submit", "finish", "complete",
                        "confirm", "place order", "pay",
                    )
                )
            )
            result = await mcp_click(
                x=proposed.get("x", 0),
                y=proposed.get("y", 0),
                element_id=proposed.get("element_id"),
                prevent_deselect=answer_choice,
                replay_safe=replay_safe_navigation,
            )
            if result["success"]:
                if result.get("no_op"):
                    return (
                        "→ NO-OP: answer target was already selected; click "
                        "suppressed to avoid deselecting it"
                    )
                # At-most-once controls may have accepted the native click even
                # when their custom CSS/DOM exposes no machine-readable change.
                # Do not call that a confirmed OK: keeping it explicitly pending
                # preserves the Intent Journal's do-not-repeat guard until the
                # fresh screen/DOM proves whether the effect applied.
                if result.get("verified") is False:
                    return (
                        "→ DISPATCHED ONCE [verification pending]: click was sent "
                        "but no observable effect was confirmed; automatic replay "
                        "was suppressed. Re-read the live page before repeating."
                    )
                bits = []
                if result.get("navigated"):
                    bits.append("navigated")
                if result.get("dom_changed"):
                    bits.append("DOM changed")
                if result.get("state_verified"):
                    bits.append("control state verified")
                if clean:  # terse, NO internal strategy name
                    return "→ OK" + (f" ({', '.join(bits)})" if bits else "")
                return (f"→ OK (click via {result['strategy']}"
                        f"{',' + ' navigated' if result['navigated'] else ''}"
                        f"{',' + ' DOM changed' if result['dom_changed'] else ''})")
            return _fmt_fail(clean, verb, "CLICK INEFFECTIVE", result["error"])

        elif verb == "drag_and_drop":
            from mcp_tools import mcp_drag_and_drop
            result = await mcp_drag_and_drop(
                element_id=proposed.get("element_id"),
                x=proposed.get("x", 0) or 0,
                y=proposed.get("y", 0) or 0,
                target_x=proposed.get("target_x", 0) or 0,
                target_y=proposed.get("target_y", 0) or 0,
                target_element_id=proposed.get("target_element_id"),
            )
            if result["success"]:
                return "→ OK (dragged source to destination)"
            return _fmt_fail(clean, verb, "DRAG FAILED", result["error"])

        elif verb == "type":
            from mcp_tools import mcp_type
            result = await mcp_type(
                text=proposed.get("text", ""),
                x=proposed.get("x", 0),
                y=proposed.get("y", 0),
                element_id=proposed.get("element_id"),
                force_retype=bool(
                    proposed.get("force_retype") or proposed.get("replace_existing")
                ),
            )
            if result["success"]:
                if result.get("no_op"):
                    return (
                        "→ NO-OP: input already contained the exact requested "
                        "value; no browser mutation was made"
                    )
                if clean:
                    return f"→ OK (typed {result['actual_length']} chars)"
                return f"→ OK (typed {result['actual_length']} chars via {result['strategy']})"
            return _fmt_fail(clean, verb, "TYPE FAILED", result["error"])

        elif verb == "set_date_of_birth":
            from mcp_tools import mcp_set_date_of_birth
            result = await mcp_set_date_of_birth(
                proposed.get("element_id"), proposed.get("text", "") or ""
            )
            if result["success"]:
                return "→ OK (date-of-birth widget completed)"
            return _fmt_fail(clean, verb, "DATE INPUT FAILED", result["error"])

        elif verb == "scroll":
            from mcp_tools import mcp_scroll
            result = await mcp_scroll(600)
            if not result["success"]:
                return _fmt_fail(clean, verb, "FAILED", result["error"])
            moved = int(result.get("scrolled_px", 0) or 0)
            if abs(moved) < 5:
                return "→ NO-OP (page did not move — already at bottom or unscrollable)"
            if result.get("at_bottom"):
                return f"→ OK (scrolled {moved}px, reached page bottom)"
            return f"→ OK (scrolled {moved}px)"

        elif verb == "press_enter":
            from mcp_tools import mcp_press_enter
            result = await mcp_press_enter()
            return "→ OK" if result["success"] else _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "hover":
            from mcp_tools import mcp_hover
            result = await mcp_hover(element_id=proposed.get("element_id"),
                                     x=proposed.get("x", 0), y=proposed.get("y", 0))
            return "→ OK (hovered)" if result["success"] else _fmt_fail(clean, verb, "HOVER FAILED", result["error"])

        elif verb == "select_option":
            from mcp_tools import mcp_select_option
            result = await mcp_select_option(proposed.get("element_id"), proposed.get("text", "") or "")
            if result["success"]:
                return f"→ OK (selected '{result.get('selected', '')}')"
            return _fmt_fail(clean, verb, "SELECT FAILED", result["error"])

        elif verb == "press_key":
            from mcp_tools import mcp_press_key
            result = await mcp_press_key(proposed.get("text", "") or proposed.get("key", ""))
            return "→ OK" if result["success"] else _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "press_combo":
            from mcp_tools import mcp_press_key
            result = await mcp_press_key(
                proposed.get("key_combo", "") or proposed.get("text", "")
            )
            return "→ OK" if result["success"] else _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "scroll_to":
            from mcp_tools import mcp_scroll_directional
            result = await mcp_scroll_directional(
                proposed.get("direction", "down") or "down",
                proposed.get("scroll_amount", 500) or 500,
            )
            return "→ OK" if result["success"] else _fmt_fail(clean, verb, "FAILED", result["error"])

        elif verb == "upload_file":
            from mcp_tools import mcp_upload_file
            result = await mcp_upload_file(
                proposed.get("element_id"), proposed.get("file_path", "") or ""
            )
            return "→ OK" if result["success"] else _fmt_fail(clean, verb, "UPLOAD FAILED", result["error"])

        elif verb == "wait":
            from mcp_tools import mcp_wait
            result = await mcp_wait(proposed.get("wait_ms", 800))
            return f"→ OK (waited {result.get('waited_ms', 800)}ms)"

        else:
            return f"→ UNKNOWN verb: {verb}"

    except Exception as e:
        logger.warning("Action execution failed: %s", e)
        return f"→ CRASHED: {str(e)[:100]}"
