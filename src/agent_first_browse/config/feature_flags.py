"""Feature switches for optional cognition, perception, and recovery behavior.

Each optional capability is additive and independently gated. The master switch
`COGNITIVE_FEATURES_ENABLED=0` disables the optional layer while preserving the
deterministic execution path; individual switches can disable one capability
without affecting the others.

Defaults are ON (the features are meant to be exercised in live testing), but a
flag is honored the instant it is set — `REALITY_MONITOR_ENABLED=0` disables just the Reality
Monitor, etc. The env is read live (not cached) so a flag can be flipped between
runs without code changes.

Pattern mirrors `consensus.consensus_enabled()` (env-var truthiness) for
consistency with the existing codebase.
"""

from __future__ import annotations

import os

_OFF = ("0", "false", "no", "off")


def _flag(name: str, default: bool = True) -> bool:
    """Read a boolean env flag. Unset → `default`; any of 0/false/no/off → False."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _OFF


# ── Master switch ──────────────────────────────────────────────────────────────
def cognitive_features_enabled() -> bool:
    """Return whether optional cognitive features are enabled."""
    return _flag("COGNITIVE_FEATURES_ENABLED", True)


# ── Screen-Reality Reconciliation ────────────────────────────────────────────────
def reality_enabled() -> bool:
    """The Reality Monitor: block 'blind execution' when the live screen
    contradicts the worker's predicted change."""
    return cognitive_features_enabled() and _flag("REALITY_MONITOR_ENABLED", True)


def reality_llm_enabled() -> bool:
    """Deterministic-first: a cheap LLM reconcile call is made ONLY for genuinely
    ambiguous deltas (status==UNCLEAR). `REALITY_LLM_ENABLED=0` keeps it purely
    deterministic."""
    return reality_enabled() and _flag("REALITY_LLM_ENABLED", True)


# ── Clarity-Triggered Consensus & Vision ─────────────────────────────────────────
def clarity_consensus_enabled() -> bool:
    """Broaden PRE-action consensus from IRREVERSIBLE-only to ANY low-clarity step
    (zero risk: if unsure, poll the ensemble before acting)."""
    return cognitive_features_enabled() and _flag("CLARITY_CONSENSUS_ENABLED", True)


def target_lock_enabled() -> bool:
    """Strict goal-binding: bind the action to the target item's identity and
    resist identical-looking distractor controls (anti context-drift)."""
    return cognitive_features_enabled() and _flag("TARGET_LOCK_ENABLED", True)


def intent_journal_enabled() -> bool:
    """Atomic Intent Journaling: write-ahead record of a side-effecting action
    BEFORE it runs, fed to the next decision so a timed-out/crashed action is never
    blindly repeated (handoff-amnesia / double-toggle fix)."""
    return cognitive_features_enabled() and _flag("INTENT_JOURNAL_ENABLED", True)


def subgoal_lock_enabled() -> bool:
    """Sub-Goal Lock: a verified-complete sub-goal stays LOCKED even when the
    Outcome Judge globally rejects a premature 'done' — so the agent never re-does
    a finished sub-goal (the multi-part 'amnesia loop' fix)."""
    return cognitive_features_enabled() and _flag("SUBGOAL_LOCK_ENABLED", True)


# ── Progress-Aware Loops & Smart Scrolling ───────────────────────────────────────
def stagnation_enabled() -> bool:
    """Revive the dead `same_url_streak` into a generalized stagnation detector."""
    return cognitive_features_enabled() and _flag("STAGNATION_DETECTION_ENABLED", True)


def smart_scroll_enabled() -> bool:
    """Scroll with feedback (delta / at-bottom / new-content) instead of blind 600px."""
    return cognitive_features_enabled() and _flag("SMART_SCROLL_ENABLED", True)


# ── Page-Subject Understanding ───────────────────────────────────────────────────
def page_context_enabled() -> bool:
    """Page archetype + instruction-aware DOM re-rank."""
    return cognitive_features_enabled() and _flag("PAGE_CONTEXT_ENABLED", True)


# ── Adaptive Perception Engine (universal; P0 = Tier-1 passthrough router) ───────
def adaptive_perception_enabled() -> bool:
    """Route page perception through the Adaptive Perception Engine. P0 is a
    behavior-identical Tier-1 passthrough; later phases add deep-sweep / vision
    tiers behind the same flag. Off ⇒ the direct single-pass snapshot."""
    return cognitive_features_enabled() and _flag("ADAPTIVE_PERCEPTION_ENABLED", True)


def strict_viewport_enabled() -> bool:
    """AP-P1: strict viewport filter on Tier-1 output — drop off-screen noise;
    preserve + tag off-screen actionables (universal recall). Gated under the
    perception engine; ON by default."""
    return adaptive_perception_enabled() and _flag("STRICT_VIEWPORT_ENABLED", True)


# ── Stabilizers (DOM diffing + hybrid primitives) ────────────────────────────────
def diffing_enabled() -> bool:
    """ProgressCritic page-signal-vector diff: catch subtle overlay/modal/panel changes
    a node-count diff misses, and emit a unified state_change_score."""
    return cognitive_features_enabled() and _flag("PERCEPTION_DIFFING_ENABLED", True)


def hybrid_primitives_enabled() -> bool:
    """Clean action feedback (asymmetric verbosity + FailureClass, no strategy-name
    leak) + expanded primitives (hover / select_option / press_key)."""
    return cognitive_features_enabled() and _flag("HYBRID_PRIMITIVES_ENABLED", True)


# ── Simulator + Autonomy (flags reserved; not currently wired) ──────────────────
def webdreamer_enabled() -> bool:
    """Predictive top-K action simulation (LLM-imagined, no real actions),
    gated behind the Clarity Gate + a cost gate so it only fires on high-stakes
    ambiguous/irreversible steps."""
    return cognitive_features_enabled() and _flag("WEBDREAMER_ENABLED", True)


def webdreamer_situational_enabled() -> bool:
    """Situational scoring for WebDreamer — reward engaging a just-revealed toggle,
    decay dead-end scroll, elevate goto when stuck. `=0` ⇒ instant fallback to the
    vacuum-scoring baseline. Gated under WebDreamer."""
    return webdreamer_enabled() and _flag("WEBDREAMER_SITUATIONAL_ENABLED", True)


def lats_enabled() -> bool:
    """Tree-search/backtracking over checkpoints (default OFF until built)."""
    return cognitive_features_enabled() and _flag("LATS_ENABLED", False)


def skill_memory_enabled() -> bool:
    """Deepened retrieval-augmented procedural memory (default OFF until built)."""
    return cognitive_features_enabled() and _flag("SKILL_MEMORY_ENABLED", False)


def active_flags() -> dict[str, bool]:
    """Snapshot of every flag — logged once at startup for run-log auditability."""
    return {
        "COGNITIVE_FEATURES_ENABLED": cognitive_features_enabled(),
        "REALITY_MONITOR_ENABLED": reality_enabled(),
        "REALITY_LLM_ENABLED": reality_llm_enabled(),
        "CLARITY_CONSENSUS_ENABLED": clarity_consensus_enabled(),
        "TARGET_LOCK_ENABLED": target_lock_enabled(),
        "INTENT_JOURNAL_ENABLED": intent_journal_enabled(),
        "SUBGOAL_LOCK_ENABLED": subgoal_lock_enabled(),
        "STAGNATION_DETECTION_ENABLED": stagnation_enabled(),
        "SMART_SCROLL_ENABLED": smart_scroll_enabled(),
        "PAGE_CONTEXT_ENABLED": page_context_enabled(),
        "ADAPTIVE_PERCEPTION_ENABLED": adaptive_perception_enabled(),
        "STRICT_VIEWPORT_ENABLED": strict_viewport_enabled(),
        "PERCEPTION_DIFFING_ENABLED": diffing_enabled(),
        "HYBRID_PRIMITIVES_ENABLED": hybrid_primitives_enabled(),
        "WEBDREAMER_ENABLED": webdreamer_enabled(),
        "WEBDREAMER_SITUATIONAL_ENABLED": webdreamer_situational_enabled(),
        "LATS_ENABLED": lats_enabled(),
        "SKILL_MEMORY_ENABLED": skill_memory_enabled(),
    }
