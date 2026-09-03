"""Brain Graph — LangGraph StateGraph Orchestration Spine for True Brain v16.0.

This is the replacement for the monolithic for-loop in advanced_agent.py.
It wires together: goal compiler, planner, perceiver, router, worker nodes,
overwatch verifier, commit/rollback/recovery nodes, and the finalizer.

Key architectural properties:
  1. TYPED STATE: BrainState flows through every node
  2. CHECKPOINTING: SqliteSaver persists state at every super-step
  3. CONDITIONAL EDGES: MoE routing + verdict-based branching
  4. CYCLES: retry/rollback/replan loops with bounded iteration
  5. AUDIT TRAIL: every step is logged and replayable

The graph does NOT spawn multiple agents. It uses a single LLM failover
chain with SPECIALIZED PROMPTS per worker node. This follows the "earn
your cost" principle: multi-agent complexity only when genuinely needed.

References:
  - LangGraph docs: StateGraph, conditional edges, checkpointing
  - MAST (arXiv 2503.13657): typed state defeats system-design failures
  - Six Sigma Agent (arXiv 2601.22290): checkpoint-every-k-steps strategy
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Ensure app imports work
sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

from brain_state import BrainState
from moe_router import route_to_worker, verdict_router
import mcp_tools

try:
    from app.logger import get_logger
    logger = get_logger("brain_graph")
except ImportError:
    logger = logging.getLogger("brain_graph")


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-Level State (initialized at runtime)
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE = None
_CONTEXT = None
_FAILOVER_CHAIN = []   # full text chain — auxiliary calls (planner, PRM, judge)
_WORKER_CHAIN = []     # V24 role separation — worker action decisions (top tier only)
_AUXILIARY_CHAIN = []  # high-volume support calls — provider-prioritized
_VISION_CHAIN = []
_AUDIO_CHAIN = []
_BREAKER = None
_HEALTH_TRACKER = None
_INVOKE_FN = None
_CRITIC = None
_DREAMER = None
_PRM_CRITIC = None
_SKILL_MEM = None
_DISPLAY_SETUP_ACTIVE = False


async def _sync_active_page(fallback_url: str = "", *, force_recovery: bool = False):
    """Synchronize perception with tab handoffs and recover dead renderers."""
    global _PAGE, _CRITIC
    if _PAGE is None:
        return None
    try:
        recovery = await mcp_tools.recover_unusable_page(
            _PAGE, fallback_url=fallback_url, force=force_recovery
        )
        if recovery.get("recovered"):
            active = recovery.get("page")
        else:
            await mcp_tools.adopt_new_page_if_opened(_PAGE)
            active = mcp_tools.get_page()
    except Exception as exc:
        logger.debug("Tab synchronization skipped (non-fatal): %s", exc)
        return _PAGE

    if active is not _PAGE:
        _PAGE = active
        if _CRITIC is not None:
            try:
                _CRITIC._page = active
            except Exception:
                pass
        try:
            from site_customizations import apply_current_site_customizations
            await apply_current_site_customizations(active)
        except Exception:
            pass
        logger.info("🧭 Perception now follows active tab: %s", getattr(active, "url", "")[:120])
    return _PAGE


async def _wait_for_perception_readiness(page, state: BrainState) -> None:
    """Wait for useful browser state without sleeping a fixed amount per step."""
    if page is None:
        return
    try:
        from survey_context import survey_perception_wait_mode
        mode = survey_perception_wait_mode(state.model_dump(), getattr(page, "url", ""))
    except Exception:
        mode = "standard"

    if mode == "standard":
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            await page.wait_for_timeout(1500)
        except Exception as exc:
            if mcp_tools.is_page_crash_error(exc):
                raise
            pass
        return

    load_timeout_ms = max(10000, int(os.getenv("SURVEY_LOAD_TIMEOUT_MS", "30000")))
    if mode == "navigation":
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=load_timeout_ms)
        except Exception:
            # Client-side routers and redirect chains do not always emit a clean
            # load-state event; the readiness signal below remains authoritative.
            pass
        settle_budget_ms = max(1500, int(os.getenv("SURVEY_LOAD_WAIT_MS", "8000")))
        # Paidwork uses client-side route transitions where the shell appears
        # before the survey cards. Require a longer, stable observation window
        # so fast-path navigation cannot click back into the shell mid-hydration.
        if "paidwork.com" in str(getattr(page, "url", "")).lower():
            settle_budget_ms = max(settle_budget_ms, 5000)
    else:
        settle_budget_ms = max(
            150,
            min(1000, int(os.getenv("SURVEY_SAME_PAGE_SETTLE_MS", "500"))),
        )

    deadline = time.monotonic() + settle_budget_ms / 1000.0
    poll_interval = max(
        0.05,
        min(0.25, int(os.getenv("SURVEY_READINESS_POLL_MS", "100")) / 1000.0),
    )
    # The click/type layer already performed post-action verification. On an
    # ordinary same-page action one useful sample is sufficient; navigation
    # still requires two matching samples to avoid reading mid-hydration.
    stable_required = 0 if (
        mode == "same_page" and str(state.action_outcome or "").startswith("→ OK")
    ) else 1
    if "paidwork.com" in str(getattr(page, "url", "")).lower() and mode == "navigation":
        stable_required = 2
    last_signature = ""
    stable_samples = 0
    while time.monotonic() < deadline:
        try:
            signal = await page.evaluate(r"""() => {
                const body = document.body;
                const text = (body?.innerText || '').replace(/\s+/g, ' ').trim();
                const visible = [...document.querySelectorAll(
                    'button,input,select,textarea,a[href],[role="button"],[role="radio"],[role="checkbox"]'
                )].filter(el => {
                    const r = el.getBoundingClientRect();
                    const s = getComputedStyle(el);
                    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none';
                }).length;
                const busy = [...document.querySelectorAll(
                    '[aria-busy="true"],progress,.loading,.loader,.spinner,[class*="loading"],[class*="spinner"]'
                )].some(el => {
                    const r = el.getBoundingClientRect();
                    const s = getComputedStyle(el);
                    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none';
                });
                const loading = busy || /^(loading|please wait|redirecting)\b|\bloading\.\.\.$/i.test(text.slice(0, 180));
                return {
                    ready: document.readyState !== 'loading',
                    useful: visible > 0 || text.length >= 120,
                    loading,
                    signature: `${document.readyState}|${visible}|${text.length}|${text.slice(0, 100)}`,
                };
            }""")
            signature = str(signal.get("signature") or "")
            stable_samples = stable_samples + 1 if signature == last_signature else 0
            last_signature = signature
            if (
                signal.get("ready")
                and signal.get("useful")
                and not signal.get("loading")
                and stable_samples >= stable_required
            ):
                return
        except Exception as exc:
            if mcp_tools.is_page_crash_error(exc):
                raise
            # Execution contexts can disappear during redirects. Keep polling
            # within the bounded navigation budget instead of failing the run.
            stable_samples = 0
            last_signature = ""
        # A renderer-independent sleep cannot itself crash when the target dies.
        await asyncio.sleep(poll_interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Goal Compiler
# ═══════════════════════════════════════════════════════════════════════════════

async def goal_compiler_node(state: BrainState) -> dict:
    """Parse objective, set up authorization gates, extract domain."""
    objective = state.objective

    # Auto-allow domains from objective
    from execution_safety import auto_allow_from_objective
    auto_allow_from_objective(objective)

    # Extract target domain
    domain = ""
    url_match = re.search(r'https?://(?:www\.)?([\\w.-]+)', objective)
    if url_match:
        domain = url_match.group(1)

    logger.info("🎯 Goal compiled: %s", objective[:120])
    logger.info("🌐 Target domain: %s", domain or "(none)")

    from survey_context import is_continuous_survey_mission
    continuous_survey_mode = is_continuous_survey_mission(objective)
    if continuous_survey_mode:
        logger.info("♾️ Continuous survey mode active — one completion is a cycle, not termination")

    return {
        "task_domain": domain,
        "continuous_survey_mode": continuous_survey_mode,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Planner
# ═══════════════════════════════════════════════════════════════════════════════

async def planner_node(state: BrainState) -> dict:
    """Strategic planner — forms an APPROACH + finish-line + steps in one call.

    V18: instead of just a list of checkpoints, the planner now produces a
    StrategicPlan{strategy, success_criteria, assumptions, steps}. The strategy
    and assumptions seed the agent's persistent cognition (so it reasons WITH a
    theory rather than re-deriving one every step), and success_criteria gives
    the agent an explicit finish line (anti done-confusion).
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from cognition import StrategicPlan, merge_beliefs

    objective = state.objective

    sys_msg = SystemMessage(content=(
        "You are a senior web-automation strategist. Given an objective, produce: "
        "(1) a short overall STRATEGY/approach, (2) the SUCCESS CRITERIA describing "
        "exactly what the page shows when the task is done, (3) 2-4 testable "
        "ASSUMPTIONS, and (4) 3-5 sequential STEPS (each one browser action). "
        "For survey/questionnaire objectives, NEVER plan to select the first, top, "
        "or a random answer. Plan to read each question and ground the answer in its "
        "literal instruction, the active respondent profile, or objective reasoning."
    ))
    user_msg = HumanMessage(content=f"Objective: {objective}")

    strategy = ""
    success_criteria = ""
    assumptions: list[str] = []
    checkpoints: list[str] = []
    try:
        plan, _ = await _INVOKE_FN(
            _AUXILIARY_CHAIN or _FAILOVER_CHAIN, [sys_msg, user_msg], StrategicPlan,
            _BREAKER, health_tracker=_HEALTH_TRACKER,
        )
        strategy = (getattr(plan, "strategy", "") or "").strip()
        success_criteria = (getattr(plan, "success_criteria", "") or "").strip()
        assumptions = [a.strip() for a in (getattr(plan, "assumptions", []) or []) if a.strip()]
        checkpoints = [s.strip() for s in (getattr(plan, "steps", []) or []) if s.strip()]
    except Exception as e:
        logger.warning("Strategic planner failed: %s — using objective as single step", e)
    if not checkpoints:
        checkpoints = [objective]

    try:
        from survey_context import sanitize_survey_plan
        strategy, checkpoints = sanitize_survey_plan(objective, strategy, checkpoints)
    except Exception as e:  # noqa: BLE001
        logger.debug("Survey plan sanitization skipped (non-fatal): %s", e)

    if state.continuous_survey_mode:
        strategy = (
            (strategy + " ") if strategy else ""
        ) + (
            "Treat each credited or completed survey as one cycle: return to the dashboard, "
            "select the next best reward-per-minute offer, and continue until the user stops the run."
        )
        success_criteria = (
            "Continuous session: there is no autonomous terminal page. After each completion/credit, "
            "the agent returns to the survey dashboard and begins the next survey; only user interruption ends it."
        )
        repeat_step = "After completion or credit, return to the survey dashboard and repeat with the next best offer"
        if not any("repeat" in step.lower() or "next survey" in step.lower() for step in checkpoints):
            checkpoints.append(repeat_step)

    # Build plan steps
    plan_steps = []
    for i, desc in enumerate(checkpoints[:7]):
        plan_steps.append({
            "id": i,
            "desc": desc.strip(),
            "status": "active" if i == 0 else "pending",
        })

    # Seed cognition: strategy + bounded initial beliefs (from assumptions)
    beliefs = merge_beliefs([], assumptions)
    if strategy:
        logger.info("🧭 Strategy: %s", strategy[:160])
    if success_criteria:
        logger.info("🏁 Done when: %s", success_criteria[:120])

    # Generate PRM checklist
    prm_checklist = []
    if _PRM_CRITIC:
        try:
            prm_checklist_items = await _PRM_CRITIC.generate_checklist(objective)
            prm_checklist = [{"desc": c.description, "status": "pending"}
                            for c in prm_checklist_items]
            logger.info("📋 PRM Checklist (%d items) generated", len(prm_checklist))
        except Exception as e:
            logger.warning("PRM checklist generation failed: %s", e)

    # Retrieve skill context
    skill_context = ""
    if _SKILL_MEM:
        try:
            domain = state.task_domain
            relevant = _SKILL_MEM.retrieve_relevant(objective, domain=domain)
            skill_context = _SKILL_MEM.inject_into_prompt(relevant)
            if skill_context:
                logger.info("🧠 SkillMemory: %d workflows injected", len(relevant))
        except Exception as e:
            logger.warning("SkillMemory retrieval failed: %s", e)

    logger.info("📋 Plan (%d steps): %s", len(plan_steps),
                " → ".join(s["desc"][:40] for s in plan_steps))

    return {
        "plan_steps": plan_steps,
        "plan_cursor": 0,
        "plan_progress_pct": 0,
        "prm_checklist": prm_checklist,
        "skill_context": skill_context,
        # ── Cognition seed ──
        "strategy": strategy,
        "success_criteria": success_criteria,
        "beliefs": beliefs,
        "strategy_confidence": 1.0,
        "current_obstacle": "",
        "ladder_rung": 0,
        "tried_tactics": [],
        "goal_score_window": [],
        "restrategize_count": 0,
        "goal_complete_hint": "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Perceive — DOM extraction + page FSM update
# ═══════════════════════════════════════════════════════════════════════════════

async def perceive_node(state: BrainState) -> dict:
    """Extract DOM, detect login state, update page FSM."""
    recovery_url = state.survey_home_url if state.continuous_survey_mode else ""
    page = (
        await _sync_active_page(recovery_url)
        if recovery_url else await _sync_active_page()
    )

    # A continuous survey uses a rolling action window rather than a terminal
    # budget. This check also recovers a resumed checkpoint exactly at its limit.
    rolling_budget_update: dict[str, int] = {}
    if state.step_number >= state.max_steps:
        if state.continuous_survey_mode:
            from survey_context import rolling_continuous_budget
            rolling_budget_update["max_steps"] = rolling_continuous_budget(
                state.step_number, state.max_steps
            )
            logger.info("♾️ Continuous survey budget rolled forward to %d",
                        rolling_budget_update["max_steps"])
        else:
            return {"next_node": "finalize"}

    # Long grace periods are now reserved for true navigation/load events.
    # Ordinary same-page survey actions settle in a bounded sub-second window.
    try:
        await _wait_for_perception_readiness(page, state)
    except Exception as exc:
        if not mcp_tools.is_page_crash_error(exc):
            raise
        logger.warning("🛟 Renderer crashed during perception; restoring the survey dashboard")
        page = await _sync_active_page(recovery_url, force_recovery=True)
        if page is None:
            raise
        await _wait_for_perception_readiness(page, state)

    current_url = page.url
    effective_max_steps = rolling_budget_update.get("max_steps", state.max_steps)
    logger.info("━━━ Step %d/%d ━━━", state.step_number + 1, effective_max_steps)
    display_url = current_url
    if state.continuous_survey_mode or "survey" in state.objective.lower():
        try:
            from survey_context import compact_survey_url
            display_url = compact_survey_url(current_url)
        except Exception:
            display_url = current_url[:360]
    logger.info("URL: %s", display_url)

    # DOM snapshot — routed through the Adaptive Perception Engine when enabled.
    # P0: the engine is a behavior-identical Tier-1 passthrough over this same
    # mcp_snapshot call; it adds the routing seam for later deep-sweep / vision
    # tiers without changing this node. A hard fallback to the direct snapshot
    # guarantees perception can never be broken by the engine layer.
    async def _direct_snapshot() -> dict:
        return await mcp_tools.mcp_snapshot()

    snapshot = None
    try:
        from feature_flags import adaptive_perception_enabled
        if adaptive_perception_enabled():
            from perception_engine import perceive as _perceive
            pr = await _perceive(page, ctx={
                "objective": state.objective,
                "bound_target": state.bound_target,
                "step_number": state.step_number,
            })
            snapshot = {
                "elements": pr.elements, "markdown": pr.markdown,
                "page_text": pr.page_text,
                "selector_map": pr.selector_map, "element_count": pr.element_count,
                "sparse_dom_status": getattr(pr, "sparse_dom_status", "NOT_NEEDED"),
                "sparse_dom_control_count": getattr(pr, "sparse_dom_control_count", 0),
                "sparse_dom_reason": getattr(pr, "sparse_dom_reason", ""),
                "paidwork_selection_ready": getattr(pr, "paidwork_selection_ready", None),
            }
    except Exception as e:
        logger.warning("Adaptive perception failed (%s) — using direct snapshot", e)
        snapshot = None
    if snapshot is None:
        snapshot = await _direct_snapshot()

    elements_list = snapshot.get("elements", [])
    dom_markdown = snapshot.get("markdown", "")
    page_text = snapshot.get("page_text", "")
    selector_map = snapshot.get("selector_map", {})
    element_count = snapshot.get("element_count", 0)
    sparse_dom_status = str(snapshot.get("sparse_dom_status") or "NOT_NEEDED")
    sparse_dom_reason = str(snapshot.get("sparse_dom_reason") or "")
    paidwork_selection_ready = snapshot.get("paidwork_selection_ready")
    paidwork_selection_waits = int(state.paidwork_selection_waits or 0)
    if paidwork_selection_ready is False:
        paidwork_selection_waits = (
            paidwork_selection_waits + 1
            if current_url == state.current_url else 1
        )
    else:
        paidwork_selection_waits = 0
    dom_recovery_attempts = state.dom_recovery_attempts
    if sparse_dom_status != "NOT_NEEDED":
        dom_recovery_attempts += 1
        logger.info(
            "🧩 DOM recovery audit %s (attempt=%d controls=%s reason=%s)",
            sparse_dom_status, dom_recovery_attempts,
            snapshot.get("sparse_dom_control_count", 0), sparse_dom_reason or "none",
        )
        if sparse_dom_status == "UNRESOLVED":
            try:
                from survey_context import survey_failure_fingerprint
                logger.warning(
                    "🧾 MANUAL_REVIEW sparse_dom url_key=%s",
                    survey_failure_fingerprint(
                        current_url, kind="sparse_dom", reason=sparse_dom_reason
                    ),
                )
            except Exception:
                pass

    survey_provider_urls = list(state.survey_provider_urls or [])
    if state.continuous_survey_mode and not survey_provider_urls:
        try:
            from survey_context import survey_provider_urls as configured_survey_providers
            survey_provider_urls = configured_survey_providers()
        except Exception:
            survey_provider_urls = []
    survey_provider_index = min(
        max(0, int(state.survey_provider_index or 0)),
        max(0, len(survey_provider_urls) - 1),
    )
    configured_provider_home = (
        survey_provider_urls[survey_provider_index] if survey_provider_urls else ""
    )
    survey_home_url = state.survey_home_url or configured_provider_home
    empty_page_streak = state.survey_empty_page_streak
    if state.continuous_survey_mode and current_url != survey_home_url:
        empty_page_streak = empty_page_streak + 1 if not page_text.strip() and not selector_map else 0
    else:
        empty_page_streak = 0
    survey_offers_present = False
    if state.continuous_survey_mode:
        try:
            from survey_context import rank_survey_offers
            survey_offers_present = bool(rank_survey_offers(selector_map))
            if survey_offers_present:
                survey_home_url = current_url
        except Exception:
            pass

    # Dashboard navigation is a separate stall class from an unchanged survey
    # question. Offer lists often mutate enough to reset same_url/fingerprint
    # tracking while the agent keeps clicking inert cards for dozens of turns.
    # Count dashboard perception turns independently and rotate before that can
    # consume the whole run.
    dashboard_stall_steps = int(state.survey_dashboard_stall_steps or 0)
    dashboard_stall_since = float(state.survey_dashboard_stall_since or 0.0)
    if survey_offers_present:
        dashboard_stall_steps += 1
        if dashboard_stall_since <= 0:
            dashboard_stall_since = time.time()
    else:
        dashboard_stall_steps = 0
        dashboard_stall_since = 0.0

    survey_profile = state.survey_profile
    survey_profile_render = state.survey_profile_render
    if state.continuous_survey_mode or "survey" in state.objective.lower():
        try:
            from survey_profile import (
                compact_runtime_profile,
                load_active_profile,
                render_profile,
            )
            durable_profile = load_active_profile()
            survey_profile = compact_runtime_profile(durable_profile, page_text)
            survey_profile_render = render_profile(durable_profile, page_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Question-relevant profile refresh skipped (non-fatal): %s", exc)

    survey_fingerprint = ""
    survey_interaction_fingerprint = ""
    survey_page_advanced = False
    survey_question_transitions = state.survey_question_transitions
    survey_cycles_completed = state.survey_cycles_completed
    completion_signature = state.last_survey_completion_signature
    cycle_boundary_pending = state.survey_cycle_boundary_pending
    cycle_context_reset = False
    cycle_cleanup_updates: dict = {}
    completion = ""
    if state.continuous_survey_mode or "survey" in state.objective.lower():
        try:
            from survey_context import (
                survey_completion_evidence,
                survey_interaction_fingerprint as interaction_fingerprint_for,
                survey_page_fingerprint,
            )

            survey_fingerprint = survey_page_fingerprint(page_text, selector_map)
            survey_interaction_fingerprint = interaction_fingerprint_for(selector_map)
            from survey_context import is_verified_survey_page_transition
            # Dynamic dashboards/widgets can mutate after a click that the
            # executor explicitly reported as failed. That churn is not a
            # survey-question transition and must not reset stagnation or make
            # the PRM claim that live questions are advancing.
            survey_page_advanced = is_verified_survey_page_transition(
                state.survey_page_fingerprint,
                survey_fingerprint,
                state.action_outcome,
                previous_url=state.current_url,
                current_url=current_url,
                current_page_text=page_text,
                action=state.proposed_action,
            )
            if survey_page_advanced:
                survey_question_transitions += 1
                logger.info("📈 Survey page advanced (%d transitions)", survey_question_transitions)

            completion = survey_completion_evidence(page_text)
            if completion:
                new_signature = f"{current_url}|{survey_fingerprint}"
                if not completion_signature:
                    survey_cycles_completed += 1
                    logger.info("🏁 Survey cycle %d completed — continuing", survey_cycles_completed)
                    cycle_boundary_pending = True
                completion_signature = new_signature
            else:
                # Reset after leaving completion so an identical provider page
                # can count as a fresh completion on the next cycle.
                completion_signature = ""
                if cycle_boundary_pending:
                    from survey_context import survey_cycle_cleanup_updates
                    cycle_context_reset = True
                    cycle_boundary_pending = False
                    cycle_cleanup_updates = survey_cycle_cleanup_updates({
                        **state.model_dump(),
                        "survey_cycles_completed": survey_cycles_completed,
                    })
                    logger.info(
                        "🧹 Survey cycle boundary: compacted local context; "
                        "durable profile and lifetime counters preserved"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Survey progress fingerprint skipped (non-fatal): %s", exc)

    # Provider lifecycle and the conservative same-question wall-clock guard.
    # This is layered on the existing Overwatch/commit ledger: model claims,
    # volatile query tokens, and checkbox/input churn do not reset it. Only a
    # changed canonical route or question resets the hard same-question clock.
    provider_question_started = state.survey_provider_question_started
    provider_rotate_required = False
    abandon_required = False
    boundary_reason = ""
    boundary_target_url = ""
    unsupported_routes = list(state.survey_unsupported_routes or [])
    unsupported_kind = ""
    stuck_watch_updates: dict[str, Any] = {}
    if state.continuous_survey_mode:
        try:
            from survey_context import (
                should_rotate_survey_provider,
                survey_failure_kind,
                unsupported_survey_requirement,
                canonical_survey_url,
                survey_dashboard_stall_step_limit,
                survey_dashboard_stall_timeout_seconds,
                survey_stuck_watch_updates,
            )

            failure_kind = survey_failure_kind(page_text)
            unsupported_kind = unsupported_survey_requirement(page_text)
            if unsupported_kind:
                route_key = canonical_survey_url(current_url)
                if route_key and route_key not in unsupported_routes:
                    unsupported_routes.append(route_key)
                if state.survey_offer_signature:
                    unsupported_offer_signatures = list(
                        state.survey_unsupported_offer_signatures or []
                    )
                    if state.survey_offer_signature not in unsupported_offer_signatures:
                        unsupported_offer_signatures.append(state.survey_offer_signature)
                else:
                    unsupported_offer_signatures = list(
                        state.survey_unsupported_offer_signatures or []
                    )
            else:
                unsupported_offer_signatures = list(
                    state.survey_unsupported_offer_signatures or []
                )
            # Some providers expose terminal screen-outs in the route while
            # rendering little or no accessible text. Handle those immediately
            # instead of spending vision budget or waiting for stagnation.
            if not failure_kind and not unsupported_kind and re.search(
                r"/(?:error-nc|disqualified|screened[-_]?out)(?:/|$)",
                current_url or "", re.IGNORECASE,
            ):
                failure_kind = "disqualified"
            # The first click commonly leaves the provider dashboard for a
            # partner/screener URL before the model has committed an answer.
            # Treat that external route as an entered survey immediately;
            # otherwise the entry timer can close a valid screener while it is
            # still loading or waiting for its first Continue.
            external_survey_route = bool(
                current_url
                and current_url not in {survey_home_url, configured_provider_home}
                and current_url.startswith(("http://", "https://"))
            )
            if external_survey_route and not failure_kind:
                provider_question_started = True
            if (
                not failure_kind
                and not unsupported_kind
                and not page_text.strip()
                and not selector_map
                and empty_page_streak >= 2
            ):
                failure_kind = "load_failed"

            active_survey_page = bool(
                not survey_offers_present
                and current_url
                and (
                    provider_question_started
                    or current_url not in {survey_home_url, configured_provider_home}
                )
                and not completion
                and not failure_kind
                and not unsupported_kind
            )
            stuck_watch_updates = survey_stuck_watch_updates(
                {
                    **state.model_dump(),
                    "survey_provider_question_started": provider_question_started,
                },
                current_url=current_url,
                page_fingerprint=survey_fingerprint,
                interaction_fingerprint=survey_interaction_fingerprint,
                active=active_survey_page,
            )

            dashboard_stalled = (
                survey_offers_present
                and (
                    dashboard_stall_steps >= survey_dashboard_stall_step_limit()
                    or (
                        dashboard_stall_since > 0
                        and time.time() - dashboard_stall_since
                        >= survey_dashboard_stall_timeout_seconds()
                    )
                )
                and current_url == (survey_home_url or configured_provider_home)
            )
            fresh_completion_dashboard = ""
            if completion:
                try:
                    from survey_site_quirks import fresh_dashboard_after_completion
                    fresh_completion_dashboard = fresh_dashboard_after_completion(
                        survey_home_url or configured_provider_home
                    )
                except Exception:
                    fresh_completion_dashboard = ""

            if fresh_completion_dashboard:
                abandon_required = True
                boundary_reason = "completed:fresh_dashboard"
                boundary_target_url = fresh_completion_dashboard
                logger.info(
                    "🗂️ Verified completion requires a fresh provider dashboard tab: %s",
                    fresh_completion_dashboard,
                )
            elif dashboard_stalled:
                if survey_provider_urls and len(survey_provider_urls) > 1:
                    from survey_outcomes import choose_survey_provider_index
                    next_provider_index = choose_survey_provider_index(
                        survey_provider_urls,
                        current_index=survey_provider_index,
                        pending_failure=True,
                        exclude_current=True,
                    )
                    boundary_reason = "provider_rotated:dashboard_navigation_stall"
                    boundary_target_url = survey_provider_urls[next_provider_index]
                    logger.warning(
                        "🧭 Dashboard navigation stalled for %d turns; rotating provider %d → %d",
                        dashboard_stall_steps, survey_provider_index, next_provider_index,
                    )
                else:
                    boundary_reason = "abandoned:dashboard_navigation_stall"
                    boundary_target_url = survey_home_url or configured_provider_home
                    logger.warning(
                        "🧭 Dashboard navigation stalled for %d turns; refreshing dashboard",
                        dashboard_stall_steps,
                    )
                abandon_required = True
            elif unsupported_kind and current_url != (survey_home_url or configured_provider_home):
                abandon_required = True
                boundary_reason = f"unsupported_capability:{unsupported_kind}"
                boundary_target_url = survey_home_url or configured_provider_home
                logger.warning(
                    "🚫 Unsupported survey capability=%s; abandoning route=%s",
                    unsupported_kind, canonical_survey_url(current_url)[:140],
                )
            elif failure_kind and current_url != (survey_home_url or configured_provider_home):
                abandon_required = True
                boundary_reason = (
                    f"screened_out:{failure_kind}"
                    if failure_kind in {"disqualified", "quota_full"}
                    else f"abandoned:{failure_kind}"
                )
                if survey_provider_urls:
                    from survey_outcomes import choose_survey_provider_index
                    next_provider_index = choose_survey_provider_index(
                        survey_provider_urls,
                        current_index=survey_provider_index,
                        pending_failure=True,
                    )
                    boundary_target_url = survey_provider_urls[next_provider_index]
                else:
                    boundary_target_url = survey_home_url or configured_provider_home
            elif stuck_watch_updates.get("survey_stuck_timed_out"):
                abandon_required = True
                boundary_reason = "abandoned:unchanged_page_timeout"
                if survey_provider_urls:
                    from survey_outcomes import choose_survey_provider_index
                    next_provider_index = choose_survey_provider_index(
                        survey_provider_urls,
                        current_index=survey_provider_index,
                        pending_failure=True,
                    )
                    boundary_target_url = survey_provider_urls[next_provider_index]
                else:
                    boundary_target_url = survey_home_url or configured_provider_home
                logger.warning(
                    "⏱️ Survey question unchanged for %.1fs; abandoning",
                    stuck_watch_updates.get("survey_stuck_elapsed_seconds", 0.0),
                )
            else:
                provider_rotate_required = should_rotate_survey_provider({
                    **state.model_dump(),
                    "survey_provider_question_started": provider_question_started,
                })
                if provider_rotate_required and survey_provider_urls:
                    from survey_outcomes import choose_survey_provider_index
                    next_provider_index = choose_survey_provider_index(
                        survey_provider_urls,
                        current_index=survey_provider_index,
                        pending_failure=True,
                        exclude_current=True,
                    )
                    abandon_required = True
                    boundary_reason = "provider_rotated:entry_step_limit"
                    boundary_target_url = survey_provider_urls[next_provider_index]
                    logger.warning(
                        "🔁 Provider %d exhausted its entry budget after %d committed steps; "
                        "rotating to provider %d",
                        survey_provider_index,
                        max(0, state.step_number - state.survey_provider_start_step),
                        next_provider_index,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Survey provider lifecycle check skipped (non-fatal): %s", exc)
    if cycle_context_reset:
        provider_question_started = False
        provider_rotate_required = False
        abandon_required = False
        boundary_reason = ""
        boundary_target_url = ""

    # Audio/video animal challenges get one bounded media-analysis call. A first
    # inaccessible capture asks the worker to click Play; the next perception
    # retries once, then makes a constrained non-'none' guess if still unavailable.
    survey_audio_analysis = dict(state.survey_audio_analysis or {})
    survey_audio_key = state.survey_audio_challenge_key
    survey_audio_attempts = int(state.survey_audio_attempts or 0)
    try:
        from survey_audio import audio_challenge_key, analyze_audio_challenge

        detected_key = audio_challenge_key(current_url, page_text, selector_map)
        if not detected_key:
            survey_audio_analysis = {}
            survey_audio_key = ""
            survey_audio_attempts = 0
        else:
            if detected_key != survey_audio_key:
                survey_audio_analysis = {}
                survey_audio_key = detected_key
                survey_audio_attempts = 0
            terminal = survey_audio_analysis.get("status") in {"identified", "guessed"}
            if not terminal and survey_audio_attempts < 2:
                survey_audio_analysis = await analyze_audio_challenge(
                    page,
                    url=current_url,
                    page_text=page_text,
                    selector_map=selector_map,
                    audio_chain=_AUDIO_CHAIN,
                    invoke_fn=_INVOKE_FN,
                    health_tracker=_HEALTH_TRACKER,
                    allow_guess_without_capture=(survey_audio_attempts >= 1),
                )
                survey_audio_attempts += 1
                if survey_audio_analysis:
                    logger.info(
                        "🔊 Survey audio attempt %d → %s",
                        survey_audio_attempts,
                        survey_audio_analysis.get("status", "unknown"),
                    )
    except Exception as exc:  # noqa: BLE001 - audio support must never stop perception
        logger.warning("Survey audio analysis skipped (non-fatal): %s", str(exc)[:140])

    # Login detection
    login_result = await mcp_tools.mcp_detect_login()
    login_detected = login_result.get("logged_in", False)
    if login_detected:
        logger.info("🔑 Login detected: profile=True")

    # URL streak tracking
    same_url_streak = state.same_url_streak
    last_url = state.last_url_for_streak
    streak_url = current_url
    streak_last_url = last_url
    if state.continuous_survey_mode:
        try:
            from survey_context import canonical_survey_url
            streak_url = canonical_survey_url(current_url)
            streak_last_url = canonical_survey_url(last_url)
        except Exception:
            pass
    url_changed = streak_url != streak_last_url
    if cycle_context_reset or survey_page_advanced or url_changed:
        same_url_streak = 0
    else:
        same_url_streak += 1
    if url_changed:
        if _DREAMER:
            _DREAMER.clear_cache()

    # Learn from observed action effects, not labels. Three alternating
    # transitions (A→B, B→A, A→B) identify the exact B-side action that would
    # close the loop again. This remains element-specific so another provider's
    # identically labelled control can still be explored.
    navigation_cycle_note = ""
    navigation_cycle_blocked_action: dict[str, Any] = {}
    if not cycle_context_reset:
        try:
            from stagnation import detect_navigation_cycle
            from survey_context import survey_page_fingerprint
            nav_signal = detect_navigation_cycle(
                state.history,
                current_url=current_url,
                current_fingerprint=survey_page_fingerprint(dom_markdown, selector_map),
            )
            if nav_signal.detected:
                navigation_cycle_note = nav_signal.note
                navigation_cycle_blocked_action = nav_signal.blocked_action
                logger.warning("🔂 NAVIGATION CYCLE LEARNED: %s", nav_signal.note[:180])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Navigation-cycle learning skipped (non-fatal): %s", exc)

    # Pre-compute plan/facts/history renders
    context_state = state.model_copy(update=cycle_cleanup_updates) if cycle_cleanup_updates else state
    plan_render = context_state.get_plan_render()
    facts_render = state.render_facts()
    history_compressed = context_state.compress_history()
    try:
        from survey_context import render_cycle_answer_memory
        survey_cycle_memory_render = render_cycle_answer_memory(
            context_state.survey_cycle_answers, page_text
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Survey-cycle memory retrieval skipped (non-fatal): %s", exc)
        survey_cycle_memory_render = context_state.survey_cycle_memory_render

    # ── V29 Phase 3: progress-aware stagnation (revives the previously-DEAD
    #    same_url_streak signal). "Busy but not progressing" → break the loop. ──
    stagnation_level = 0
    stagnation_note = ""
    try:
        from feature_flags import stagnation_enabled
        if stagnation_enabled():
            from stagnation import detect_stagnation
            from survey_context import has_recent_survey_progress
            recent_survey_progress = has_recent_survey_progress({
                **state.model_dump(),
                "survey_page_advanced": survey_page_advanced,
            })
            sig = detect_stagnation({
                "same_url_streak": same_url_streak,
                "goal_score_window": state.goal_score_window,
                "loop_signatures": state.loop_signatures,
                "recent_survey_progress": recent_survey_progress,
            })
            stagnation_level = 0 if cycle_context_reset else sig.level
            stagnation_note = "" if cycle_context_reset else sig.note
            if sig.stuck:
                logger.warning("🔁 STAGNATION (level %d): %s", sig.level,
                               "; ".join(sig.reasons))
    except Exception as e:
        logger.debug("Stagnation check skipped (non-fatal): %s", e)

    return {
        **rolling_budget_update,
        **cycle_cleanup_updates,
        "current_url": current_url,
        "dom_markdown": dom_markdown,
        "page_text": page_text,
        "selector_map": selector_map,
        "elements_list": elements_list,
        "element_count": element_count,
        "dom_recovery_attempts": dom_recovery_attempts,
        "dom_recovery_status": sparse_dom_status,
        "dom_recovery_reason": sparse_dom_reason,
        "paidwork_selection_ready": paidwork_selection_ready,
        "paidwork_selection_waits": paidwork_selection_waits,
        "survey_profile": survey_profile,
        "survey_profile_render": survey_profile_render,
        "survey_page_fingerprint": survey_fingerprint,
        "survey_interaction_fingerprint": survey_interaction_fingerprint,
        "survey_page_advanced": survey_page_advanced,
        "survey_question_transitions": survey_question_transitions,
        "survey_cycles_completed": survey_cycles_completed,
        "last_survey_completion_signature": completion_signature,
        "survey_cycle_boundary_pending": cycle_boundary_pending,
        "survey_audio_analysis": survey_audio_analysis,
        "survey_audio_challenge_key": survey_audio_key,
        "survey_audio_attempts": survey_audio_attempts,
        "survey_home_url": survey_home_url,
        "survey_empty_page_streak": empty_page_streak,
        "survey_provider_urls": survey_provider_urls,
        "survey_provider_index": survey_provider_index,
        "survey_provider_question_started": provider_question_started,
        "survey_dashboard_stall_steps": dashboard_stall_steps,
        "survey_dashboard_stall_since": dashboard_stall_since,
        "survey_provider_rotate_required": provider_rotate_required,
        "survey_abandon_required": abandon_required,
        "survey_boundary_reason": boundary_reason,
        "survey_boundary_target_url": boundary_target_url,
        "survey_unsupported_routes": unsupported_routes,
        "survey_unsupported_offer_signatures": locals().get(
            "unsupported_offer_signatures",
            list(state.survey_unsupported_offer_signatures or []),
        ),
        "survey_unsupported_count": len(unsupported_routes),
        **stuck_watch_updates,
        "login_detected": login_detected,
        "page_fsm": "READY",
        "same_url_streak": same_url_streak,
        "last_url_for_streak": streak_url,
        "navigation_cycle_note": navigation_cycle_note,
        "navigation_cycle_blocked_action": navigation_cycle_blocked_action,
        # V29 stagnation signals (read by the guidance bus next worker step)
        "stagnation_level": stagnation_level,
        "stagnation_note": stagnation_note,
        # Pre-computed renders (avoids re-computing in workers)
        "plan_render": plan_render,
        "facts_render": facts_render,
        "history_compressed": history_compressed,
        "survey_cycle_memory_render": survey_cycle_memory_render,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Router (MoE)
# ═══════════════════════════════════════════════════════════════════════════════

def router_node(state: BrainState) -> dict:
    """MoE router — classifies the next worker to invoke."""
    # Check circuit breaker
    if _BREAKER and _BREAKER.tripped:
        logger.critical("🔌 CIRCUIT BREAKER: %s", _BREAKER.reason)
        if state.continuous_survey_mode:
            return {"next_node": "recovery"}
        return {"next_node": "finalize"}

    target = route_to_worker(state.model_dump())
    return {"next_node": target}


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Workers (Navigator, Interactor, Extractor)
# ═══════════════════════════════════════════════════════════════════════════════

async def navigator_worker(state: BrainState) -> dict:
    """Navigation specialist — goto, scroll, wait."""
    from workers.base_worker import invoke_worker, build_system_prompt
    prompt = build_system_prompt(
        "navigator",
        plan_context=state.get_plan_render(),
        facts_context=state.render_facts(),
        skill_context=state.skill_context,
    )
    return await invoke_worker(
        state.model_dump(), prompt,
        _WORKER_CHAIN or _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
        vision_chain=_VISION_CHAIN,
        dreamer=_DREAMER,
    )


async def interactor_worker(state: BrainState) -> dict:
    """Interaction specialist — click, type, form fill."""
    from workers.base_worker import invoke_worker, build_system_prompt
    prompt = build_system_prompt(
        "interactor",
        plan_context=state.get_plan_render(),
        facts_context=state.render_facts(),
        skill_context=state.skill_context,
    )
    return await invoke_worker(
        state.model_dump(), prompt,
        _WORKER_CHAIN or _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
        vision_chain=_VISION_CHAIN,
        dreamer=_DREAMER,
    )


async def extractor_worker(state: BrainState) -> dict:
    """Data extraction specialist — read, find, capture."""
    from workers.base_worker import invoke_worker, build_system_prompt
    prompt = build_system_prompt(
        "extractor",
        plan_context=state.get_plan_render(),
        facts_context=state.render_facts(),
        skill_context=state.skill_context,
    )
    return await invoke_worker(
        state.model_dump(), prompt,
        _WORKER_CHAIN or _FAILOVER_CHAIN, _BREAKER, _HEALTH_TRACKER, _INVOKE_FN,
        vision_chain=_VISION_CHAIN,
        dreamer=_DREAMER,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Overwatch
# ═══════════════════════════════════════════════════════════════════════════════

async def overwatch_wrapper(state: BrainState) -> dict:
    """Wrap the Overwatch verifier subgraph into a LangGraph node."""
    from overwatch import overwatch_node
    return await overwatch_node(state.model_dump(), _PAGE, _CRITIC)


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Done Check (runs Overwatch in done-mode)
# ═══════════════════════════════════════════════════════════════════════════════

async def done_check_node(state: BrainState) -> dict:
    """Special Overwatch path for 'done' actions."""
    from overwatch import overwatch_node
    return await overwatch_node(state.model_dump(), _PAGE, _CRITIC)


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Commit — Checkpoint + Advance Plan
# ═══════════════════════════════════════════════════════════════════════════════

async def _run_prm_audit(state: BrainState) -> tuple[float, list[dict], list[str]]:
    """Revived PRM goal-audit: score the live page against the checklist.

    Reconstructs ChecklistItem objects from the persisted dicts, scores them
    (PRMCritic mutates statuses in place), and writes the statuses back.
    Returns (goal_score 0..1, updated_checklist_dicts, newly_completed_descs).
    """
    from prm_critic import ChecklistItem

    items = [
        ChecklistItem(
            id=i,
            description=d.get("desc", ""),
            status=d.get("status", "pending"),
            confidence=float(d.get("confidence", 0.0)),
            step_completed=int(d.get("step_completed", -1)),
            verified=bool(d.get("verified", False)),   # V26: persist stickiness
            evidence=d.get("evidence", ""),
        )
        for i, d in enumerate(state.prm_checklist)
    ]
    step_score = await _PRM_CRITIC.score_step(
        checklist=items,
        dom_markdown=state.dom_markdown,
        current_url=state.current_url,
        step_number=state.step_number + 1,
    )
    updated = [
        {"desc": it.description, "status": it.status,
         "confidence": it.confidence, "step_completed": it.step_completed,
         "verified": it.verified, "evidence": it.evidence}
        for it in items
    ]
    return step_score.total_score, updated, step_score.newly_completed


async def commit_node(state: BrainState) -> dict:
    """Commit after Overwatch passes. Advance plan, extend budget if needed.

    V18: a committed step is a VERIFIED-PROGRESS step, so it reinforces strategy
    confidence and clears the transient 'stuck' state (the obstacle is resolved —
    no stale escalation directive may bleed forward). A throttled PRM goal-audit
    gives a goal-aware progress signal, detects stalls, and nudges the agent to
    finish once the goal looks achieved (anti done-confusion).
    """
    from cognition import (
        update_confidence, prm_should_audit, push_goal_score, detect_stall,
        clear_transient, DONE_CEILING,
    )

    updates = {
        "page_fsm": "READY",
        "step_number": state.step_number + 1,
        "error_count": 0,
        "recovery_count": 0,
        "retry_count": 0,
        "session_checkpoint_id": f"step_{state.step_number}",
        "proposed_action": None,
        "overwatch_verdict": "",
    }

    # Reinforce confidence (verified progress) + clear transient stuck state.
    updates["strategy_confidence"] = update_confidence(state.strategy_confidence, progress=True)
    updates.update(clear_transient())

    if state.continuous_survey_mode:
        from survey_context import rolling_continuous_budget
        rolled_max = rolling_continuous_budget(state.step_number, state.max_steps)
        if rolled_max > state.max_steps:
            updates["max_steps"] = rolled_max
            updates["budget_extended"] = True
            logger.info("♾️ Continuous survey budget rolled forward to %d", rolled_max)

        # Extend the existing verified-submission ledger. This runs only after
        # Overwatch has reality-checked and passed the action, so model intent or
        # an uncommitted retry can never reset the 180-second same-page timer.
        committed_action = state.proposed_action or {}
        committed_verb = str(committed_action.get("verb") or "")
        committed_target = str(committed_action.get("target_name") or "").lower()
        committed_basis = str(committed_action.get("answer_basis") or "").lower()
        committed_question = str(committed_action.get("question_text") or "").strip()
        action_outcome = str(state.action_outcome or "")
        action_ok = action_outcome.lstrip().startswith("→ OK")
        offer_dashboard_visible = False
        try:
            from survey_context import rank_survey_offers
            offer_dashboard_visible = bool(rank_survey_offers(state.selector_map or {}))
        except Exception:
            pass
        away_from_dashboard = bool(
            state.current_url
            and (
                state.current_url != state.survey_home_url
                or not offer_dashboard_visible
            )
        )
        answer_action = bool(
            action_ok
            and away_from_dashboard
            and committed_verb in {
                "click", "type", "select_option", "press_enter", "press_key",
                "drag_and_drop",
            }
            and (
                (committed_question and committed_basis not in {"page_navigation", "reward_per_minute"})
                or any(term in committed_target for term in ("next", "continue", "submit"))
            )
        )
        if answer_action:
            updates["survey_provider_question_started"] = True
        last_history = list(state.history or [])[-1:] or [{}]
        verified_transition = bool(
            last_history[0].get("survey_transition_verified")
            or last_history[0].get("survey_completion_verified")
            or state.survey_page_advanced
        )
        if answer_action and verified_transition:
            verified_step = state.step_number + 1
            updates["survey_verified_progress_step"] = verified_step
            # A verified answer/transition is real progress even when an SPA
            # retains the same route and question-container identity. Reset the
            # hard clock here so slow inference cannot cause the next perception
            # to abandon immediately after a successful field interaction.
            updates["survey_stuck_since"] = time.time()
            updates["survey_stuck_page_identity"] = ""
            updates["survey_stuck_timed_out"] = False
            updates["survey_stuck_elapsed_seconds"] = 0.0
            logger.info(
                "✅ Verified survey question transition committed at action step %d",
                verified_step,
            )

        if (
            action_ok
            and committed_basis == "reward_per_minute"
            and committed_verb == "click"
        ):
            reward_value = str(committed_action.get("offer_reward") or "")
            reward_currency = str(committed_action.get("offer_currency") or "")
            updates["survey_offer_reward"] = reward_value
            updates["survey_offer_currency"] = reward_currency
            updates["survey_offer_id"] = str(committed_action.get("element_id") or "")
            updates["survey_offer_signature"] = re.sub(
                r"\s+", " ", str(committed_action.get("target_name") or "")
            ).strip().lower()[:180]
            try:
                updates["survey_offer_minutes"] = float(
                    committed_action.get("offer_minutes") or 0.0
                )
            except (TypeError, ValueError):
                updates["survey_offer_minutes"] = 0.0

        if committed_verb == "abandon_survey" and action_ok:
            from survey_context import survey_cycle_cleanup_updates

            reason = str(
                committed_action.get("survey_boundary_reason")
                or state.survey_boundary_reason
                or "abandoned"
            )
            target_url = str(committed_action.get("url") or state.survey_boundary_target_url or "")
            try:
                from survey_outcomes import get_survey_outcome_store
                providers = list(state.survey_provider_urls or [])
                current_panel = (
                    state.survey_home_url
                    or (
                        providers[state.survey_provider_index]
                        if providers and state.survey_provider_index < len(providers)
                        else ""
                    )
                )
                reward = ""
                if state.survey_offer_reward:
                    reward = f"{state.survey_offer_currency}{state.survey_offer_reward}"
                provider_started_at = float(state.survey_provider_started_at or 0.0)
                elapsed_seconds = (
                    max(1.0, time.time() - provider_started_at)
                    if provider_started_at > 0
                    else min(300.0, max(1.0, float(state.step_number + 1) * 5.0))
                )
                cycle = get_survey_outcome_store().record(
                    panel_url=current_panel,
                    result=reason,
                    elapsed_seconds=elapsed_seconds,
                    survey_url=state.current_url,
                    questions=max(
                        0,
                        state.survey_question_transitions
                        - state.survey_provider_start_transitions,
                    ),
                    reward=reward,
                    offer_minutes=(state.survey_offer_minutes or None),
                )
                if cycle:
                    logger.info(
                        "📊 Survey outcome recorded: panel=%s survey=%s result=%s elapsed=%.1fs",
                        cycle.get("panel_host") or "unknown",
                        cycle.get("survey_host") or "unknown",
                        cycle.get("result") or reason,
                        cycle.get("elapsed_seconds") or 0.0,
                    )
            except Exception as exc:
                logger.warning("Survey outcome telemetry failed (non-fatal): %s", exc)
            cleanup = survey_cycle_cleanup_updates(
                {**state.model_dump(), "step_number": state.step_number + 1},
                outcome=reason,
            )
            updates.update(cleanup)
            providers = list(state.survey_provider_urls or [])
            if target_url in providers:
                updates["survey_provider_index"] = providers.index(target_url)
            updates["survey_home_url"] = target_url or state.survey_home_url
            updates["survey_provider_start_step"] = state.step_number + 1
            updates["survey_provider_started_at"] = time.time()
            if reason.startswith("screened_out:"):
                updates["survey_screened_out_count"] = state.survey_screened_out_count + 1
            elif reason.startswith("abandoned:") or reason == "abandoned":
                updates["survey_abandoned_count"] = state.survey_abandoned_count + 1
            logger.info(
                "🧹 Survey boundary committed (%s); local context reset #%d",
                reason,
                updates.get("survey_context_resets", state.survey_context_resets),
            )

    # Check mission success
    if state.mission_success:
        if state.continuous_survey_mode:
            logger.warning("♾️ Ignoring terminal success latch during continuous survey mode")
            updates["mission_success"] = False
            updates["done_evidence"] = ""
        else:
            updates["next_node"] = "finalize"
            return updates

    # Plan advancement — check if current action indicates progress
    plan_steps = list(state.plan_steps)
    plan_cursor = state.plan_cursor
    advanced = False

    for i, step in enumerate(plan_steps):
        if step.get("status") in ("active", "in_progress"):
            # V17: semantic plan advancement — the action outcome must indicate
            # success AND the action should relate to the current plan step.
            # Old heuristic (`"OK" in outcome`) advanced on ANY success, causing
            # critical steps to be skipped prematurely.
            outcome = state.action_outcome or ""
            action_ok = ("OK" in outcome or "navigated" in outcome.lower()
                         or "SUCCESS" in outcome.upper())
            if action_ok:
                # Only advance if the action plausibly matches this plan step.
                # We check overlap between action context and step description.
                step_desc = step.get("desc", "").lower()
                action_ctx = " ".join([
                    (state.proposed_action or {}).get("verb", ""),
                    (state.proposed_action or {}).get("target_name", ""),
                    (state.proposed_action or {}).get("text", "") or "",
                    outcome,
                ]).lower()

                # Advance if any meaningful keyword from the step appears in
                # the action context, OR if the step is a generic navigation
                # step that any successful action can satisfy.
                step_words = {w for w in step_desc.split() if len(w) > 3}
                overlap = step_words & {w for w in action_ctx.split() if len(w) > 3}
                is_generic = any(kw in step_desc for kw in
                                 ("navigate", "open", "go to", "visit", "load"))

                if overlap or is_generic:
                    plan_steps[i] = {**step, "status": "done"}
                    # Activate next pending step
                    for j in range(i + 1, len(plan_steps)):
                        if plan_steps[j].get("status") == "pending":
                            plan_steps[j] = {**plan_steps[j], "status": "active"}
                            plan_cursor = j
                            break
                    advanced = True
            break

    if advanced:
        done_count = sum(1 for s in plan_steps if s.get("status") == "done")
        progress_pct = int(done_count / len(plan_steps) * 100) if plan_steps else 0
        updates["plan_steps"] = plan_steps
        updates["plan_cursor"] = plan_cursor
        updates["plan_progress_pct"] = progress_pct

        # Dynamic budget extension
        if (not state.continuous_survey_mode
                and state.step_number >= state.max_steps - 3
                and progress_pct >= 60):
            updates["max_steps"] = state.max_steps + 5
            updates["budget_extended"] = True
            logger.info("📊 Budget extended to %d (progress=%d%%)",
                        updates["max_steps"], progress_pct)

    # ── Throttled PRM goal-audit (revived): goal-aware progress + stall + done-nudge ──
    prm_audit_due = prm_should_audit(state.step_number, advanced)
    if state.continuous_survey_mode:
        try:
            audit_every = max(5, int(os.getenv("SURVEY_PRM_AUDIT_TRANSITIONS", "25")))
        except (TypeError, ValueError):
            audit_every = 25
        # The survey has its own mechanical transition/completion ledger. A
        # goal-scoring model call after every small answer adds latency without
        # improving safety, so sample only real page transitions periodically.
        prm_audit_due = bool(
            state.survey_page_advanced
            and state.survey_question_transitions > 0
            and state.survey_question_transitions % audit_every == 0
        )
    if (_PRM_CRITIC and state.prm_checklist and prm_audit_due):
        try:
            goal_score, updated_checklist, newly = await _run_prm_audit(state)
            updates["prm_checklist"] = updated_checklist
            window = push_goal_score(state.goal_score_window, goal_score)
            updates["goal_score_window"] = window

            # V26: make the SUCCESS verification first-class in the logs (was only
            # the struggle that showed up). Each newly-verified sub-goal is logged
            # with its captured on-page proof and marked sticky.
            prev_verified = {d.get("desc") for d in state.prm_checklist if d.get("verified")}
            for d in updated_checklist:
                if d.get("verified") and d.get("desc") not in prev_verified:
                    logger.info("✅ VERIFIED [%s] — %s",
                                (d.get("desc") or "")[:50],
                                (d.get("evidence") or "state change confirmed")[:120])

            # Done-ceiling nudge — reduce post-completion wandering.
            # CoVe (Overwatch L4) still guards against a PREMATURE done.
            if goal_score >= DONE_CEILING and not state.continuous_survey_mode:
                updates["goal_complete_hint"] = (
                    "\n\n✅ GOAL APPEARS COMPLETE (checklist ~satisfied). Verify your "
                    "DONE WHEN criteria on screen and output action_type='done' unless "
                    "something is clearly still missing. Do NOT keep acting once done."
                )
            else:
                updates["goal_complete_hint"] = ""

            # Stall: DOM keeps changing (we committed) but goal-score is flat →
            # busy-but-not-progressing → penalize confidence to provoke a revision.
            from survey_context import has_recent_survey_progress
            recent_survey_progress = (
                state.continuous_survey_mode
                and has_recent_survey_progress(state.model_dump())
            )
            if detect_stall(window) and not recent_survey_progress:
                penalized = update_confidence(updates["strategy_confidence"], progress=False)
                logger.info("📉 Goal-progress STALL (score window flat) — confidence %.2f→%.2f",
                            updates["strategy_confidence"], penalized)
                updates["strategy_confidence"] = penalized
            elif detect_stall(window) and recent_survey_progress:
                logger.info("📈 PRM flat score ignored — live survey questions are advancing")
        except Exception as e:
            logger.debug("PRM audit error (non-fatal): %s", e)

    # ── V17 Win-State Recognizer (generalized — zero hardcoded platforms) ──
    # If the agent just executed an IRREVERSIBLE action (as classified by
    # action_classifier: add-to-cart, submit, fork, star, post, etc.) AND the
    # page responded with a significant state change (URL changed or large DOM
    # delta), this is the cognitive signal that the CRITICAL task action just
    # completed. Inject a strong nudge to call 'done' immediately — preventing
    # the agent from wandering into post-action pages (cart recommendations,
    # confirmation screens, etc.) and clicking similar-looking buttons.
    #
    # This is NOT hardcoding any platform. The action_classifier already knows
    # what "critical" looks like from its regex patterns (buy, submit, add to
    # cart, delete, post, etc.) — all task-agnostic. The URL change is a
    # universal signal that the action had a real effect.
    action = state.proposed_action or {}
    if (not state.continuous_survey_mode
            and action.get("risk_level") == "IRREVERSIBLE"
            and state.action_outcome
            and "OK" in state.action_outcome):
        # The critical action succeeded — check if the page reacted
        outcome_lower = state.action_outcome.lower()
        page_reacted = ("navigated" in outcome_lower
                        or "structure changed" in outcome_lower
                        or "element count changed" in outcome_lower)
        if page_reacted:
            logger.info(
                "🏆 WIN-STATE: IRREVERSIBLE action '%s' on '%s' succeeded with "
                "page reaction — nudging agent to verify and finish.",
                action.get("verb", "?"), action.get("target_name", "?")[:40],
            )
            updates["goal_complete_hint"] = (
                "\n\n🏆 CRITICAL ACTION COMPLETED: You just performed the KEY "
                "irreversible action for this task and the page confirmed it "
                "(URL/DOM changed). Your task is very likely DONE. Verify the "
                "result on screen and output action_type='done' IMMEDIATELY. "
                "Do NOT click any other buttons, links, or recommendations. "
                "Do NOT navigate away. Just verify and finish."
            )

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Retry
# ═══════════════════════════════════════════════════════════════════════════════

async def retry_node(state: BrainState) -> dict:
    """Retry after a soft failure (grounding reject, no progress)."""
    return {
        "retry_count": state.retry_count + 1,
        "proposed_action": None,  # Clear for re-evaluation
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Rollback
# ═══════════════════════════════════════════════════════════════════════════════

async def rollback_node(state: BrainState) -> dict:
    """Hard-stuck handler. The escalation ladder (in Overwatch) has already tried
    the cheap tactics; rollback decides whether to RE-STRATEGIZE.

    V18: instead of a full replan (which wiped progress), rollback either
      (a) RE-STRATEGIZES — one LLM call for a genuinely different approach,
          updating `strategy` + adding the lesson as a belief, resetting the
          ladder, then continuing (→ perceive); or
      (b) lets the ladder's current directive stand and continues (→ perceive).
    Bounded by MAX_RESTRATEGIZE; the existing error_count→recovery→finalize path
    still guarantees termination.
    """
    from cognition import (
        restrategize, needs_restrategy, merge_beliefs,
        RESTRATEGIZE_TACTIC, LADDER,
    )

    updates: dict = {
        "error_count": state.error_count + 1,
        "page_fsm": "READY",
        "proposed_action": None,
        "retry_count": 0,
        "next_node": "perceive",  # continue — no destructive full replan
    }

    ladder_exhausted = (
        RESTRATEGIZE_TACTIC in state.tried_tactics
        or state.ladder_rung >= len(LADDER)
    )
    do_restrategy = ladder_exhausted or needs_restrategy(
        state.strategy_confidence, state.restrategize_count
    )

    if do_restrategy:
        logger.warning("🔄 ROLLBACK → RE-STRATEGIZE (#%d, confidence=%.2f)",
                       state.restrategize_count + 1, state.strategy_confidence)
        new_strategy, lesson = await restrategize(
            _INVOKE_FN, _AUXILIARY_CHAIN or _FAILOVER_CHAIN,
            _BREAKER, _HEALTH_TRACKER,
            objective=state.objective,
            current_strategy=state.strategy,
            beliefs=state.beliefs,
            current_url=state.current_url,
            recent_failure=state.action_outcome,
            plan_render=state.get_plan_render(),
        )
        if new_strategy:
            updates["strategy"] = new_strategy
            updates["strategy_confidence"] = 0.6  # fresh hypothesis — moderate prior
            updates["restrategize_count"] = state.restrategize_count + 1
            if lesson:
                updates["beliefs"] = merge_beliefs(state.beliefs, [lesson])
                from brain_state import REFLECTION_MAX, append_bounded
                updates["reflections"] = append_bounded(
                    state.reflections,
                    [f"Re-strategized: {lesson[:120]}"],
                    REFLECTION_MAX,
                )
        # Reset the ladder for the new strategy + clear stale directives
        updates["current_obstacle"] = ""
        updates["ladder_rung"] = 0
        updates["tried_tactics"] = []
        updates["correction_context"] = ""
        updates["recovery_advice"] = ""
    else:
        # Ladder still has untried rungs — keep Overwatch's directive, continue.
        logger.warning("🔄 ROLLBACK at step %d — continuing with next escalation tactic",
                       state.step_number)
        from brain_state import REFLECTION_MAX, append_bounded
        updates["reflections"] = append_bounded(state.reflections, [
            f"Step {state.step_number} stuck (verdict={state.overwatch_verdict}); "
            "escalating tactic."
        ], REFLECTION_MAX)

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Recovery
# ═══════════════════════════════════════════════════════════════════════════════

async def recovery_node(state: BrainState) -> dict:
    """Recovery after too many errors — reset counters and continue."""
    logger.warning("🔧 Recovery node: resetting error state after %d errors", state.error_count)
    if state.continuous_survey_mode and _BREAKER and _BREAKER.tripped:
        # Avoid a hot graph loop while the breaker is OPEN. Its state machine
        # automatically becomes probe-eligible; the continuous run stays alive.
        wait_seconds = 5.0
        try:
            opened_at = float(getattr(_BREAKER, "_opened_at", 0.0) or 0.0)
            open_wait = float(getattr(_BREAKER, "_open_wait_secs", wait_seconds) or wait_seconds)
            wait_seconds = max(0.5, min(5.0, open_wait - (time.monotonic() - opened_at)))
        except Exception:
            pass
        logger.info("♾️ Continuous run waiting %.1fs for model breaker recovery", wait_seconds)
        await asyncio.sleep(wait_seconds)
    return {
        "error_count": 0,
        "recovery_count": state.recovery_count + 1,
        "correction_failures": 0,
        "retry_count": 0,
        "recovery_advice": "Agent recovered. Trying fresh approach.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE: Finalize
# ═══════════════════════════════════════════════════════════════════════════════

async def finalize_node(state: BrainState) -> dict:
    """Clean shutdown — record workflow, close browser."""
    mission_success = state.mission_success and not state.continuous_survey_mode
    done_evidence = state.done_evidence

    # ── V20 last-chance outcome audit: shutting down WITHOUT a verified done
    #    (budget exhausted etc.) — check the final page once before reporting
    #    failure. The goal may be achieved even though 'done' was never accepted.
    if (not state.continuous_survey_mode
            and not mission_success and not done_evidence and _PAGE is not None):
        try:
            from overwatch import final_outcome_audit
            achieved, evidence = await final_outcome_audit(state.model_dump(), _PAGE)
            done_evidence = evidence
            if achieved:
                mission_success = True
                logger.info("⚖️ Final audit: goal evidence found on shutdown — %s",
                            evidence[:160])
            else:
                logger.info("⚖️ Final audit: goal NOT confirmed — %s", evidence[:160])
        except Exception as e:
            logger.debug("Final outcome audit failed (non-fatal): %s", e)

    # Record in SkillMemory
    if _SKILL_MEM and state.history:
        try:
            _SKILL_MEM.record_workflow(
                objective=state.objective,
                steps=[dict(h) for h in state.history],
                success=mission_success,
                total_steps=state.step_number,
            )
            logger.info("📝 Workflow recorded in SkillMemory (success=%s)", mission_success)
        except Exception as e:
            logger.warning("SkillMemory record failed: %s", e)

    # Record in CampaignMemory
    try:
        from campaign_memory import CampaignMemory
        memory = CampaignMemory()
        memory.record_post(
            platform="web",
            title=state.objective[:120],
            content=state.objective,
            url=state.current_url,
            agent_model="brain_v16",
            steps_taken=state.step_number,
        )
    except Exception as e:
        logger.warning("CampaignMemory record failed: %s", e)

    if state.continuous_survey_mode:
        logger.info(
            "♾️ Continuous survey session ended after %d completed cycle(s); it was not auto-completed.",
            state.survey_cycles_completed,
        )
    elif mission_success:
        logger.info("✅ MISSION COMPLETE — Task finished successfully!")
        if done_evidence:
            logger.info("⚖️ Verified outcome: %s", done_evidence[:200])
    else:
        logger.info("⚠️ Agent shutting down (mission incomplete)")
        if done_evidence:
            logger.info("⚖️ Outcome status: %s", done_evidence[:200])

    # ── Clean handoff: reset stateful singletons so a fresh task starts clean ──
    # (BrainState is per-task already; these objects are reused if the process is.)
    try:
        if _CRITIC and hasattr(_CRITIC, "reset_for_task"):
            _CRITIC.reset_for_task()
        if _PRM_CRITIC is not None:
            _PRM_CRITIC._prev_score = 0.0
    except Exception as e:
        logger.debug("Cognitive reset on finalize failed (non-fatal): %s", e)

    # Clear cognitive state in the returned state for a tidy final snapshot.
    from cognition import clear_all
    # Persist the (possibly audit-corrected) outcome into the final state so the
    # runner's summary reports the verified result, not the pre-audit one.
    return {**clear_all(),
            "mission_success": mission_success,
            "done_evidence": done_evidence}


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph Construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_brain_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph."""
    g = StateGraph(BrainState)

    # ── Add Nodes ──
    g.add_node("goal_compiler", goal_compiler_node)
    g.add_node("planner", planner_node)
    g.add_node("perceive", perceive_node)
    g.add_node("router", router_node)
    g.add_node("navigator", navigator_worker)
    g.add_node("interactor", interactor_worker)
    g.add_node("extractor", extractor_worker)
    g.add_node("overwatch", overwatch_wrapper)
    g.add_node("done_check", done_check_node)
    g.add_node("commit", commit_node)
    g.add_node("retry", retry_node)
    g.add_node("rollback", rollback_node)
    g.add_node("recovery", recovery_node)
    g.add_node("finalize", finalize_node)

    # ── Linear Path: START → goal_compiler → planner → perceive ──
    g.add_edge(START, "goal_compiler")
    g.add_edge("goal_compiler", "planner")
    g.add_edge("planner", "perceive")

    # ── perceive → router ──
    g.add_edge("perceive", "router")

    # ── Router → workers (conditional) ──
    g.add_conditional_edges("router", lambda s: s.next_node, {
        "navigator": "navigator",
        "interactor": "interactor",
        "extractor": "extractor",
        "done_check": "done_check",
        "recovery": "recovery",
        "finalize": "finalize",
    })

    # ── All workers → overwatch ──
    g.add_edge("navigator", "overwatch")
    g.add_edge("interactor", "overwatch")
    g.add_edge("extractor", "overwatch")

    # ── Overwatch → conditional verdict ──
    g.add_conditional_edges("overwatch", lambda s: verdict_router(s.model_dump()), {
        "commit": "commit",
        "retry": "retry",
        "rollback": "rollback",
        "finalize": "finalize",
    })

    # ── Done check → conditional verdict ──
    g.add_conditional_edges("done_check", lambda s: verdict_router(s.model_dump()), {
        "commit": "commit",
        "retry": "retry",
        "rollback": "rollback",
        "finalize": "finalize",
    })

    # ── Commit → perceive (next step cycle) ──
    def commit_router(state: BrainState) -> str:
        if state.continuous_survey_mode:
            return "perceive"
        if state.mission_success:
            return "finalize"
        if state.step_number >= state.max_steps:
            return "finalize"
        return "perceive"

    g.add_conditional_edges("commit", commit_router, {
        "perceive": "perceive",
        "finalize": "finalize",
    })

    # ── Retry → perceive (re-perceive and re-route) ──
    g.add_edge("retry", "perceive")

    # ── Rollback → perceive (V18: continue with the re-strategy / next tactic;
    #    no destructive full replan that wipes plan progress) ──
    g.add_conditional_edges("rollback", lambda s: s.next_node or "perceive", {
        "perceive": "perceive",
        "finalize": "finalize",
    })

    # ── Recovery → perceive ──
    g.add_edge("recovery", "perceive")

    # ── Finalize → END ──
    g.add_edge("finalize", END)

    return g


# ═══════════════════════════════════════════════════════════════════════════════
#  Runner — Initialize modules and execute the graph
# ═══════════════════════════════════════════════════════════════════════════════

async def run_brain(objective: str, headless: bool = False):
    """Initialize all modules and execute the brain graph."""
    global _PAGE, _CONTEXT, _FAILOVER_CHAIN, _WORKER_CHAIN, _AUXILIARY_CHAIN
    global _VISION_CHAIN, _AUDIO_CHAIN, _BREAKER, _HEALTH_TRACKER
    global _INVOKE_FN, _CRITIC, _DREAMER, _PRM_CRITIC, _SKILL_MEM, _DISPLAY_SETUP_ACTIVE

    # ── Import existing modules ──
    from advanced_agent import (
        launch_browser, SessionGuard, shutdown_browser, _invoke_with_failover,
    )
    from model_registry import ModelRegistry
    from app.browser_promoter.browser_warmup import run_warmup, extract_target_url_from_objective
    from app.browser_promoter.worker_planner import ReasoningAgent
    from orchestrator.critic_v12 import CriticV12
    from prm_critic import PRMCritic
    from web_dreamer import WebDreamer
    from skill_memory import SkillMemory

    # CDP attaches to an already-running Windows browser, so display setup and
    # launch-time headless choices do not apply. Keep Xvfb only for the explicit
    # local Playwright fallback when no endpoint is configured.
    cdp_endpoint = os.getenv("LOCAL_CDP_ENDPOINT", "").strip()
    _DISPLAY_SETUP_ACTIVE = False
    if cdp_endpoint:
        launch_headless = False
        logger.info("🌐 Browser mode: LOCAL_CDP (%s)", cdp_endpoint)
        logger.info("🖥️ Xvfb/display setup skipped: Chrome is already running on Windows.")
    else:
        from virtual_display import start_virtual_display

        _DISPLAY_SETUP_ACTIVE = True
        launch_headless = headless
        if not headless:
            launch_headless = not start_virtual_display(1920, 1080)
        logger.info(
            "🌐 Browser mode: LOCAL_PLAYWRIGHT %s",
            "HEADLESS" if launch_headless else "HEADED (stealth)",
        )

    # ── Launch Browser ──
    _CONTEXT, _PAGE = await launch_browser(headless=launch_headless)
    guard = SessionGuard.get()

    # Initialize the registry before browser warmup so independent startup
    # work can overlap. Chains are still snapshotted only after both complete.
    registry = ModelRegistry.get_instance()
    _BREAKER = registry.breaker
    _HEALTH_TRACKER = registry.health

    # ── Warmup + model probing (independent, joined before chain construction) ──
    target_hint = extract_target_url_from_objective(objective)
    startup_results = await asyncio.gather(
        run_warmup(_PAGE, target_url=target_hint),
        registry.probe_and_prune(
            timeout=max(8.0, float(os.getenv("MODEL_PROBE_TIMEOUT_SECONDS", "15"))),
            vision_timeout=max(12.0, float(os.getenv("MODEL_VISION_PROBE_TIMEOUT_SECONDS", "25"))),
        ), return_exceptions=True,
    )
    for startup_name, startup_result in zip(("warmup", "model probe"), startup_results):
        if isinstance(startup_result, Exception):
            logger.warning("Startup %s failed (non-fatal): %s", startup_name, startup_result)

    reasoning_agent = ReasoningAgent()
    _FAILOVER_CHAIN = reasoning_agent.get_failover_chain()
    chain_names = reasoning_agent.get_chain_names()
    _INVOKE_FN = _invoke_with_failover

    if not _FAILOVER_CHAIN:
        logger.error("No LLM clients available. Check API keys in .env.")
        await shutdown_browser(_CONTEXT)
        guard.detach()
        return

    logger.info("LLM Failover Chain (%d models): %s",
                len(_FAILOVER_CHAIN), " → ".join(chain_names))
    logger.info(
        "⏱️ Model policy: ordinary failover=%ss/%s attempts; "
        "survey failover=%ss, per-model=%ss; timeout cooldown≤%ss",
        os.getenv("MODEL_FAILOVER_BUDGET_SECONDS", "15"),
        os.getenv("MODEL_FAILOVER_MAX_ATTEMPTS", "5"),
        os.getenv("SURVEY_WORKER_FAILOVER_BUDGET_SECONDS", "45"),
        os.getenv("SURVEY_WORKER_MODEL_TIMEOUT_SECONDS", "15"),
        os.getenv("MODEL_TIMEOUT_RETRY_COOLDOWN_MAX_SECONDS", "60"),
    )

    # ── V24 role separation: the worker (action decisions, the critical path)
    #    uses only the top-tier capable models; auxiliary calls use the full chain.
    _WORKER_CHAIN = registry.get_worker_chain()
    logger.info("🎯 [%s mode] Worker chain (%d top models): %s",
                registry.mode, len(_WORKER_CHAIN),
                " → ".join(registry.get_worker_chain_names()))

    # High-volume support calls prefer Google/Cloudflare/NVIDIA independently
    # of the critical worker's capability ordering.
    # These are cloned wrappers around the same clients, so health/cooldown state
    # remains shared by instance name while ordering can differ by role.
    _AUXILIARY_CHAIN = registry.get_auxiliary_chain()
    logger.info(
        "🧰 Auxiliary chain (%d models): %s",
        len(_AUXILIARY_CHAIN),
        " → ".join(registry.get_auxiliary_chain_names()),
    )

    # ── V29 Cognitive Overhaul — log the active feature switches (auditability) ──
    try:
        from feature_flags import active_flags, v29_enabled
        flags = active_flags()
        if v29_enabled():
            on = [k for k, v in flags.items() if v and k != "V29_ENABLED"]
            logger.info("🧬 V29 Cognitive Overhaul ACTIVE — %s", ", ".join(on) or "(master only)")
        else:
            logger.info("🧬 V29 disabled (V29_ENABLED=0) — running pure V28 behavior")
    except Exception as e:
        logger.debug("V29 flag log skipped (non-fatal): %s", e)

    # Clear any durable intent ledger left behind by a previously-crashed run, so a
    # stale write-ahead record can never contaminate this fresh task's audit trail.
    try:
        from intent_journal import resolve_intent
        resolve_intent()
    except Exception:
        pass

    # ── V21 Vision chain (for on-demand consults; a11y DOM stays the default) ──
    try:
        _VISION_CHAIN = registry.get_vision_chain()
        if _VISION_CHAIN:
            logger.info("👁️ Vision chain ready for on-demand consults (%d models): %s",
                        len(_VISION_CHAIN), " → ".join(registry.get_vision_chain_names()))
        else:
            logger.info("👁️ No vision models configured — agent runs a11y-DOM only")
    except Exception as e:
        _VISION_CHAIN = []
        logger.warning("Vision chain unavailable (a11y-DOM only): %s", e)

    # Audio stays dormant unless perception detects a survey media challenge.
    try:
        _AUDIO_CHAIN = registry.get_audio_chain()
        if _AUDIO_CHAIN:
            logger.info(
                "🔊 Audio chain ready for survey media (%d models): %s",
                len(_AUDIO_CHAIN),
                " → ".join(registry.get_audio_chain_names()),
            )
        else:
            logger.info("🔊 No audio model configured — media questions use attempted non-none guessing")
    except Exception as e:
        _AUDIO_CHAIN = []
        logger.warning("Audio chain unavailable (guess fallback only): %s", e)

    # ── MCP Tools ──
    mcp_tools.set_page(_PAGE)
    from survey_context import (
        is_continuous_survey_mission, requested_provider_index, survey_provider_urls,
    )
    configured_survey_provider_urls = survey_provider_urls()
    selected_survey_provider_index = 0
    if is_continuous_survey_mission(objective) and configured_survey_provider_urls:
        explicit_provider = requested_provider_index(
            objective, configured_survey_provider_urls
        )
        if explicit_provider is not None:
            selected_survey_provider_index = explicit_provider
            logger.info(
                "🎯 Explicit survey provider in objective takes priority: %s",
                configured_survey_provider_urls[selected_survey_provider_index],
            )
        else:
            try:
                from survey_outcomes import choose_survey_provider_index
                selected_survey_provider_index = choose_survey_provider_index(
                    configured_survey_provider_urls
                )
            except Exception as exc:
                logger.debug("Provider outcome ranking unavailable; using configured order: %s", exc)
        default_provider = configured_survey_provider_urls[selected_survey_provider_index]
        try:
            provider_nav = await mcp_tools.mcp_navigate(default_provider)
            if provider_nav.get("success"):
                _PAGE = mcp_tools.get_page()
                logger.info("🟢 Default survey provider ready: %s", default_provider)
            else:
                logger.warning(
                    "Default survey provider navigation failed (%s); runtime will retry/rotate",
                    provider_nav.get("error", "unknown error"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Default survey provider setup skipped: %s", exc)

    # ── CriticV12 ──
    _CRITIC = CriticV12(_PAGE)
    logger.info("🧠 CriticV12 initialized")

    # ── WebDreamer ──
    try:
        _DREAMER = WebDreamer(
            invoke_fn=_INVOKE_FN,
            failover_chain=_AUXILIARY_CHAIN or _FAILOVER_CHAIN,
            breaker=_BREAKER,
            health_tracker=_HEALTH_TRACKER,
            num_candidates=max(1, int(os.getenv("WEB_DREAMER_NUM_CANDIDATES", "1"))),
            num_simulations=max(1, int(os.getenv("WEB_DREAMER_NUM_SIMULATIONS", "1"))),
        )
        logger.info("🌙 WebDreamer initialized")
    except Exception as e:
        logger.warning("WebDreamer init failed: %s", e)

    # ── PRM Critic ──
    try:
        _PRM_CRITIC = PRMCritic(
            _INVOKE_FN,
            _AUXILIARY_CHAIN or _FAILOVER_CHAIN,
            _BREAKER,
            _HEALTH_TRACKER,
        )
        logger.info("📊 PRMCritic initialized")
    except Exception as e:
        logger.warning("PRMCritic init failed: %s", e)

    # ── V20 Outcome Judge (evidence-grounded done-gate) ──
    try:
        from overwatch import configure_outcome_judge
        configure_outcome_judge(
            _INVOKE_FN,
            _AUXILIARY_CHAIN or _FAILOVER_CHAIN,
            _BREAKER,
            _HEALTH_TRACKER,
        )
        logger.info("⚖️ Outcome judge wired into the done-gate")
    except Exception as e:
        logger.warning("Outcome judge config failed (heuristic fallback): %s", e)

    # ── SkillMemory ──
    try:
        _SKILL_MEM = SkillMemory()
        stats = _SKILL_MEM.get_stats()
        logger.info("🧠 SkillMemory: %d workflows, %d reliable",
                    stats['total_workflows'], stats['reliable_workflows'])
    except Exception as e:
        logger.warning("SkillMemory init failed: %s", e)

    # ── Build Graph ──
    graph_builder = build_brain_graph()

    # ── Compile with Checkpointer ──
    db_path = Path(__file__).parent / "persistence" / "brain_checkpoints.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        brain = graph_builder.compile(checkpointer=checkpointer)

        logger.info("🧠 Brain Graph compiled with %d nodes, checkpointer=%s",
                    len(graph_builder.nodes), type(checkpointer).__name__)

        # ── Execute ──
        continuous_session = is_continuous_survey_mission(objective)
        # LangGraph also has a graph-transition guard separate from our action
        # budget. Give continuous sessions a practically unbounded transition
        # allowance; Ctrl+C remains the intended stop mechanism.
        recursion_limit = (
            int(os.getenv("CONTINUOUS_GRAPH_RECURSION_LIMIT", "1000000"))
            if continuous_session else 1000
        )
        thread_id = f"brain_{int(time.time())}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        }
        try:
            from checkpoint_retention import prune_checkpoint_database
            retention = prune_checkpoint_database(db_path, thread_id)
            if retention["checkpoints_deleted"]:
                logger.info(
                    "🧹 Checkpoint retention removed %d old snapshots across %d expired threads",
                    retention["checkpoints_deleted"], retention["threads_deleted"],
                )
        except Exception as exc:  # noqa: BLE001 - maintenance cannot block a run
            logger.warning("Checkpoint retention skipped (non-fatal): %s", exc)
        try:
            from survey_profile import compact_runtime_profile, load_active_profile, render_profile
            durable_profile = load_active_profile()
            active_profile = compact_runtime_profile(durable_profile)
            profile_render = render_profile(durable_profile)
            logger.info("🧑 Survey respondent profile loaded: %s",
                        active_profile.get("name", "default"))
        except Exception as e:
            logger.warning("Survey profile unavailable (continuing without it): %s", e)
            active_profile, profile_render = {}, ""
        initial_state = BrainState(
            objective=objective,
            survey_profile=active_profile,
            survey_profile_render=profile_render,
            continuous_survey_mode=continuous_session,
            survey_provider_urls=configured_survey_provider_urls,
            survey_provider_index=selected_survey_provider_index,
            survey_provider_start_step=0,
            survey_provider_started_at=time.time(),
            survey_provider_start_transitions=0,
            survey_home_url=(
                configured_survey_provider_urls[selected_survey_provider_index]
                if continuous_session and configured_survey_provider_urls else ""
            ),
        )

        try:
            # Latch mission_success across the stream: each event is ONE node's
            # partial output (finalize returns {}), so reading only the last
            # event misses the success flag set earlier by overwatch/commit.
            mission_success = False
            graph_node_events = 0
            try:
                checkpoint_prune_every = max(
                    10, int(os.getenv("CHECKPOINT_PRUNE_EVERY_NODES", "40"))
                )
            except (TypeError, ValueError):
                checkpoint_prune_every = 40
            async for event in brain.astream(initial_state, config):
                graph_node_events += 1
                cycle_boundary_cleaned = False
                # Log each node execution
                for node_name, node_output in event.items():
                    if node_name != "__end__":
                        logger.debug("Node '%s' → %d updates",
                                    node_name, len(node_output) if isinstance(node_output, dict) else 0)
                    if isinstance(node_output, dict) and node_output.get("mission_success"):
                        mission_success = True
                    if isinstance(node_output, dict) and "survey_context_resets" in node_output:
                        cycle_boundary_cleaned = True

                if cycle_boundary_cleaned or graph_node_events % checkpoint_prune_every == 0:
                    try:
                        from checkpoint_retention import prune_checkpoint_database
                        retention = prune_checkpoint_database(db_path, thread_id)
                        if retention["checkpoints_deleted"]:
                            logger.debug(
                                "Checkpoint retention removed %d redundant snapshots",
                                retention["checkpoints_deleted"],
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Periodic checkpoint retention skipped: %s", exc)

            if continuous_session:
                print("\n" + "═" * 60)
                print("  ⚠️  Continuous survey session ended unexpectedly.")
                print("═" * 60 + "\n")
            elif mission_success:
                print("\n" + "═" * 60)
                print("  ✅  MISSION COMPLETE — Task finished successfully!")
                print("═" * 60 + "\n")
            else:
                print("\n" + "═" * 60)
                print("  ⚠️  Mission incomplete — ran out of steps or was interrupted.")
                print("═" * 60 + "\n")

        except KeyboardInterrupt:
            if continuous_session:
                logger.info("⏹️ Continuous survey session stopped by user — saving checkpoint")
            else:
                logger.warning("KeyboardInterrupt — saving session...")
        except Exception as e:
            import traceback
            logger.error("💥 Brain crash: %s", e)
            logger.error(traceback.format_exc())
        finally:
            try:
                await shutdown_browser(_CONTEXT)
            except Exception as e:
                logger.warning("Browser shutdown/detach failed: %s", e)
            guard.detach()
            if _DISPLAY_SETUP_ACTIVE:
                try:
                    from virtual_display import stop_virtual_display

                    stop_virtual_display()
                except Exception:
                    pass
