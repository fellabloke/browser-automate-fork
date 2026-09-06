"""BrainState — Typed Global State for the Agent First Browse Architecture.

Consolidates all 15+ scattered variables from advanced_agent.py into a
single Pydantic model that flows through the LangGraph StateGraph.

Design principles:
  - Every node reads and writes ONE typed state object
  - Accumulating collections are explicitly bounded before checkpointing
  - Model-facing history is a compact recent-action + survey-answer ledger
  - Page FSM makes browser lifecycle states explicit

References:
  - MAST (arXiv 2503.13657): typed state defeats "context loss" (44.2% of failures)
  - Six Sigma Agent (arXiv 2601.22290): checkpoint-every-k-steps strategy
  - LangGraph docs: reducer semantics for accumulator fields
"""

from __future__ import annotations

import os
from typing import Literal, Any

from pydantic import BaseModel, Field


def _positive_int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


HISTORY_MAX_ENTRIES = _positive_int_env("AGENT_HISTORY_MAX_ENTRIES", 48, 12)
HISTORY_RECENT_ACTIONS = _positive_int_env("AGENT_HISTORY_RECENT_ACTIONS", 8, 3)
SURVEY_ANSWER_LEDGER_MAX = _positive_int_env("SURVEY_ANSWER_LEDGER_MAX", 32, 8)
HISTORY_PROMPT_MAX_CHARS = _positive_int_env("AGENT_HISTORY_PROMPT_MAX_CHARS", 9000, 2000)
SURVEY_CYCLE_ARCHIVE_MAX = _positive_int_env("SURVEY_CYCLE_ARCHIVE_MAX", 160, 40)
LOOP_SIGNATURE_MAX = 24
REFLECTION_MAX = 6


def append_bounded(existing: list, new_items: list, limit: int) -> list:
    """Append collection updates without allowing checkpoint state to grow forever."""
    return (list(existing or []) + list(new_items or []))[-max(1, int(limit)):]


# ═══════════════════════════════════════════════════════════════════════════════
#  Action Types — Proposed and Recorded
# ═══════════════════════════════════════════════════════════════════════════════

class ProposedAction(BaseModel):
    """An action proposed by a worker node. Never committed until Overwatch validates."""

    verb: Literal[
        "goto", "click", "type", "scroll", "press_enter", "wait", "done",
        "hover", "select_option", "press_key", "press_combo", "drag_and_drop",
        "upload_file", "scroll_to", "set_date_of_birth", "abandon_survey",
    ] = "wait"
    element_id: str | None = None
    target_name: str = ""
    text: str | None = None
    key_combo: str | None = None
    file_path: str | None = None
    direction: str | None = None
    scroll_amount: int | None = None
    url: str | None = None
    x: float | None = None
    y: float | None = None
    rationale: str = ""
    reversible: bool = True
    risk_level: str = "REVERSIBLE"  # REVERSIBLE | CAUTIOUS | IRREVERSIBLE

    # OTA chain-of-thought fields (from AutonomousAction)
    screen_state: str = ""
    previous_action_result: str = ""
    goal_progress: str = ""
    reasoning: str = ""

    def describe(self) -> str:
        """Human-readable description for logging."""
        parts = [self.verb]
        if self.element_id:
            parts.append(f"[{self.element_id}]")
        if self.target_name:
            parts.append(f"'{self.target_name[:30]}'")
        if self.text:
            parts.append(f"text='{self.text[:40]}'")
        if self.url:
            parts.append(f"url='{self.url[:50]}'")
        return " ".join(parts)


class StepRecord(BaseModel):
    """Immutable record of a completed step (appended through bounded history)."""

    step_id: int = 0
    action_verb: str = ""
    action_element_id: str | None = None
    action_text: str | None = None
    action_target_name: str = ""
    pre_state_hash: int = 0
    post_state_hash: int = 0
    verifier_verdict: Literal["pass", "fail", "retry", "rollback"] = "pass"
    outcome: str = ""
    screen_hint: str = ""
    url: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  BrainState — The single source of truth
# ═══════════════════════════════════════════════════════════════════════════════

class BrainState(BaseModel):
    """Typed global state for the LangGraph orchestration spine.

    Every node in the graph reads from and writes to this object.
    LangGraph's checkpointer serializes it at each super-step.

    Collection writers append explicitly through ``append_bounded``. This makes
    reset-at-boundary semantics possible and prevents infinite checkpoint growth.
    """

    # ── Identity & Goal ──
    objective: str = ""
    run_id: str = ""
    survey_attempt_id: str = ""
    task_domain: str = ""  # auto-extracted domain (e.g., "amazon.in")

    # ── Plan (from PlanState) ──
    plan_steps: list[dict] = Field(default_factory=list)
    plan_cursor: int = 0
    plan_progress_pct: int = 0
    reflections: list[str] = Field(default_factory=list)

    # ── Cognition Core — the agent's working theory of the task ──
    # NOTE: `beliefs` deliberately uses overwrite (not the `add` reducer) so the
    # updating node can keep it capped/de-duplicated — bounded memory prevents
    # prompt overload / hallucination. See cognition.merge_beliefs().
    strategy: str = ""                  # overall approach the agent is testing
    success_criteria: str = ""          # what "done" looks like (anti done-confusion)
    strategy_confidence: float = 1.0    # decays on no-progress, reinforces on progress
    beliefs: list[str] = Field(default_factory=list)   # bounded learned facts
    current_obstacle: str = ""          # "<url>|<plan_step>" identity for the ladder
    ladder_rung: int = 0                # escalation position for current obstacle
    tried_tactics: list[str] = Field(default_factory=list)  # tactics used on it
    goal_score_window: list[float] = Field(default_factory=list)  # PRM goal-scores
    restrategize_count: int = 0         # bounded by cognition.MAX_RESTRATEGIZE
    goal_complete_hint: str = ""        # set when PRM says the goal looks achieved

    # ── Working Memory ──
    history: list[dict] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)
    survey_profile: dict[str, Any] = Field(default_factory=dict)
    survey_profile_render: str = ""
    survey_audio_analysis: dict[str, Any] = Field(default_factory=dict)
    survey_audio_challenge_key: str = ""
    survey_audio_attempts: int = 0
    continuous_survey_mode: bool = False
    survey_page_fingerprint: str = ""
    survey_interaction_fingerprint: str = ""
    survey_page_advanced: bool = False
    survey_question_transitions: int = 0
    survey_cycles_completed: int = 0
    last_survey_completion_signature: str = ""
    survey_cycle_boundary_pending: bool = False
    survey_context_resets: int = 0
    survey_cycle_answers: list[dict] = Field(default_factory=list)
    survey_cycle_memory_render: str = ""
    survey_home_url: str = ""
    survey_failed_offers: list[str] = Field(default_factory=list)
    survey_unsupported_routes: list[str] = Field(default_factory=list)
    survey_unsupported_offer_signatures: list[str] = Field(default_factory=list)
    survey_unsupported_count: int = 0
    survey_empty_page_streak: int = 0
    survey_provider_urls: list[str] = Field(default_factory=list)
    survey_provider_index: int = 0
    survey_provider_start_step: int = 0
    survey_provider_started_at: float = 0.0
    survey_provider_start_transitions: int = 0
    survey_provider_question_started: bool = False
    # Number of consecutive perception turns spent on an offer dashboard. This
    # is independent of same_url_streak because dashboard contents can refresh
    # and change its fingerprint without any survey actually opening.
    survey_dashboard_stall_steps: int = 0
    survey_dashboard_stall_since: float = 0.0
    survey_provider_rotate_required: bool = False
    survey_verified_progress_step: int = -1
    survey_stuck_page_identity: str = ""
    survey_stuck_since: float = 0.0
    survey_stuck_progress_step: int = -1
    survey_stuck_elapsed_seconds: float = 0.0
    survey_stuck_timed_out: bool = False
    survey_model_wait_seconds: float = 0.0
    survey_action_no_effect_counts: dict[str, int] = Field(default_factory=dict)
    survey_hold_identity: str = ""
    survey_hold_count: int = 0
    survey_gate_exhausted: bool = False
    survey_abandon_required: bool = False
    survey_boundary_reason: str = ""
    survey_boundary_target_url: str = ""
    survey_last_boundary_outcome: str = ""
    survey_offer_reward: str = ""
    survey_offer_minutes: float = 0.0
    survey_offer_currency: str = ""
    survey_offer_id: str = ""
    survey_offer_signature: str = ""
    survey_abandoned_count: int = 0
    survey_screened_out_count: int = 0
    failures: list[dict] = Field(default_factory=list)
    extracted_data: list[dict] = Field(default_factory=list)

    # ── Page State / FSM ──
    page_fsm: Literal[
        "LOADING", "READY", "ACTION_PENDING", "VALIDATED", "ERROR"
    ] = "READY"
    current_url: str = ""
    dom_markdown: str = ""
    page_text: str = ""
    selector_map: dict[str, Any] = Field(default_factory=dict)
    elements_list: list[dict] = Field(default_factory=list)
    element_count: int = 0
    snapshot_revision: str = ""
    login_detected: bool = False

    # ── Control Flow ──
    step_number: int = 0
    max_steps: int = 25
    error_count: int = 0
    recovery_count: int = 0
    correction_failures: int = 0
    consecutive_identical_actions: int = 0
    last_action_signature: str = ""
    same_url_streak: int = 0
    last_url_for_streak: str = ""
    navigation_cycle_note: str = ""
    navigation_cycle_blocked_action: dict[str, Any] = Field(default_factory=dict)

    # ── Loop Detection ──
    loop_signatures: list[str] = Field(default_factory=list)

    # ── Routing & Action ──
    next_node: str = ""
    proposed_action: dict | None = None  # Serialized ProposedAction
    overwatch_verdict: str = ""  # "pass" | "retry" | "rollback" | "escalate"
    action_outcome: str = ""  # "→ OK" | "→ FAILED: ..." | etc.
    retry_count: int = 0

    # ── Vision-on-demand (a11y DOM is default; vision is a per-step toggle) ──
    vision_consults: int = 0       # per-task budget counter
    force_vision: bool = False     # set by the ladder's vision rung; consumed next step
    ineffective_streak: int = 0    # consecutive ineffective click/type on same target
    dom_recovery_attempts: int = 0
    dom_recovery_status: str = ""   # NOT_NEEDED | RECOVERED | UNRESOLVED
    dom_recovery_reason: str = ""
    paidwork_selection_ready: bool | None = None
    paidwork_selection_waits: int = 0
    captcha_read_attempts: int = 0
    captcha_comparison_attempts: int = 0
    captcha_corrections: int = 0
    captcha_refreshes: int = 0
    captcha_last_result: str = ""

    # ── Reality Monitor — screen-reality reconciliation ──
    # Written ONLY by Overwatch after executing an action. `reality_status` is
    # observational; `reality_note` carries a CONTRADICTED discrepancy back to the
    # worker via the guidance bus so the agent re-evaluates the REAL screen instead
    # of blindly proceeding. Cleared on a successful commit (clear_transient).
    reality_status: str = ""   # "" | CONFIRMED | CONTRADICTED | UNCLEAR | NULL
    reality_note: str = ""     # discrepancy fed to the guidance bus on contradiction

    # ── Unified DOM-diff state-change signal (written by Overwatch
    # from ProgressCritic). 0..1: how much the page state changed after the last action.
    state_change_score: float = 0.0

    # ── WebDreamer predictive-simulation diagnostics ──
    webdreamer_runs: int = 0       # times the dreamer simulated before acting
    webdreamer_overrides: int = 0  # times its imagined best replaced the worker pick

    # ── Clarity Gate, Target Lock, Stagnation, Smart Scroll ──
    # bound_target: the semantic identity of the current sub-task's target (written
    # by perceive_node). stagnation_*: written by perceive_node. scroll_stuck_streak:
    # written by overwatch when a scroll yields no new content / no movement.
    bound_target: str = ""             # human-readable focus (target_lock phrase)
    stagnation_level: int = 0          # 0..3 progress-signals fired
    stagnation_note: str = ""          # guidance-bus directive when stuck
    scroll_stuck_streak: int = 0       # consecutive unproductive scrolls

    # ── current Atomic Intent Journal — write-ahead record of a side-effecting action.
    # Written by Overwatch BEFORE execution; cleared on a confirmed success. While
    # set (action timed out / unconfirmed), the next worker decision is warned not
    # to blindly repeat it — fixes handoff-amnesia double-toggling.
    last_attempted_action: dict | None = None

    # ── Session Results ──
    mission_success: bool = False
    session_checkpoint_id: str | None = None
    skill_context: str = ""  # Injected SkillMemory context

    # ── PRM Checklist ──
    prm_checklist: list[dict] = Field(default_factory=list)

    # ── Recovery ──
    recovery_advice: str = ""
    correction_context: str = ""

    # ── Metrics ──
    total_actions: int = 0
    grounding_rejects: int = 0
    critic_progress: int = 0
    critic_no_progress: int = 0
    reflexion_triggers: int = 0
    done_blocked: int = 0
    # P2: consensus diagnostics — how many IRREVERSIBLE actions were put to a
    # multi-model vote, and how many were abstained (held back as too ambiguous).
    consensus_votes: int = 0
    abstentions: int = 0
    # The outcome judge's cited on-page proof (or the honest gap on an
    # unverified shutdown) — surfaced in the finalize report.
    done_evidence: str = ""

    # ── Budget extension tracking ──
    budget_extended: bool = False

    # ── Pre-computed renders (updated by perceive_node each cycle) ──
    plan_render: str = ""
    facts_render: str = ""
    history_compressed: str = ""

    def get_current_plan_step(self) -> dict | None:
        """Return the current active plan step."""
        for s in self.plan_steps:
            if s.get("status") in ("active", "in_progress"):
                return s
        return None

    def get_plan_render(self) -> str:
        """The agent's SINGLE source of cognitive state (injected into the system
        prompt). This is the ONE authoritative answer to: what is the master goal,
        what is the one sub-task to do right now, what's already done, and what to
        ignore. There is intentionally NO second/competing task list — a prior
        design that injected a separate checklist caused the agent to lose track
        of its goal and tunnel onto abstract sub-goals.
        """
        done = [s for s in self.plan_steps if s.get("status") == "done"]
        active = self.get_current_plan_step()
        pending = [s for s in self.plan_steps if s.get("status") == "pending"]

        lines = [f"🎯 MASTER GOAL: {self.objective[:240]}"]
        lines.append(f"📊 PROGRESS: {self.plan_progress_pct}% ({len(done)}/{len(self.plan_steps)} steps done)")

        if done:
            done_str = "; ".join(s["desc"][:50] for s in done[-3:])
            if len(done) > 3:
                done_str = f"...({len(done)-3} earlier); " + done_str
            lines.append(f"✓ DONE (do NOT repeat): {done_str}")

        if active:
            lines.append(
                f"▶ CURRENT SUB-TASK (your ONLY focus right now — accomplish THIS "
                f"using what is actually on the screen): {active['desc']}")
            nxt = pending[0]["desc"][:50] if pending else None
            if nxt:
                lines.append(f"   (do NOT start the next step until current is done): {nxt}")
        elif not pending:
            lines.append("▶ CURRENT SUB-TASK: all steps complete — verify the goal on screen and finish.")

        # Distraction guard — the #1 real-world failure on dense e-commerce pages.
        lines.append(
            "⚠️ STAY ON TASK: act only on the element(s) that advance your CURRENT "
            "SUB-TASK. Ignore distractions — ads, 'sponsored', 'recommended', "
            "'related products', 'people also bought', newsletter/cookie popups, "
            "and any widget unrelated to the current sub-task.")

        if self.reflections:
            lines.append(f"💡 NOTE: {self.reflections[-1][:150]}")

        return "\n".join(lines)

    def compress_history(self) -> str:
        """Render bounded context while retaining recent survey answer consistency.

        The durable respondent profile carries stable facts across cycles. This
        cycle-local ledger carries recent question/answer pairs that may be
        referenced later in a long questionnaire. Raw action history is never
        allowed to expand the model prompt indefinitely.
        """
        if not self.history:
            return "[]"
        entries = list(self.history)[-HISTORY_MAX_ENTRIES:]
        result_lines: list[str] = []
        compacted = max(0, len(self.history) - len(entries))
        if compacted:
            result_lines.append(
                f"• {compacted} older actions compacted; durable respondent facts remain in the profile."
            )

        answers = [e for e in entries if e.get("question_text") and e.get("answer_value")]
        if answers:
            result_lines.append("CURRENT SURVEY ANSWER LEDGER (cycle-local, bounded):")
            for e in answers[-SURVEY_ANSWER_LEDGER_MAX:]:
                result_lines.append(
                    f"• Q: {str(e.get('question_text') or '')[:150]} "
                    f"→ A: {str(e.get('answer_value') or '')[:100]}"
                )

        result_lines.append("RECENT VERIFIED ACTIONS:")
        for e in entries[-HISTORY_RECENT_ACTIONS:]:
            line = (f"Action turn {e.get('step', '?')}: {e.get('verb') or e.get('action', '?')} "
                    f"[{e.get('element_id') or ''}] '{e.get('target_name') or ''}' {e.get('outcome', '')}")
            if e.get("screen"):
                line += f" | Screen: {e['screen']}"
            result_lines.append(line)
        rendered = "\n".join(result_lines)
        if len(rendered) > HISTORY_PROMPT_MAX_CHARS:
            rendered = rendered[-HISTORY_PROMPT_MAX_CHARS:]
            rendered = "… earlier compact context omitted …\n" + rendered
        return rendered

    def render_facts(self) -> str:
        """Render facts for system prompt."""
        if not self.facts:
            return ""
        return "\n".join(f"• {k}: {v}" for k, v in list(self.facts.items())[-10:])
