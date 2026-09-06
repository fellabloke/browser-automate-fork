"""ModelRegistry — model-agnostic provider layer for Agent First Browse.

Enforces STRICT separation between Text, Vision, and Audio API pipelines.
No graph node should ever construct an LLM client directly.
All model access goes through this registry.

Reads ALL API keys from .env — NVIDIA NIM, Cloudflare, Google Gemini.

Responsibilities include resilient reasoning, provider health, and structured recovery:
  - ROLE-AWARE failover: the chain is sorted by explicit role priority, model
    quality tier, then expected cost. Duplicate keys for the same provider/model
    remain adjacent while empirically unreliable hosts can be demoted.
  - Startup probe: dead models (404/401) are pruned for the session;
    survivors get a measured latency seed.
  - EWMA health: per-instance latency and failure-probability estimates
    drive ordering (E[cost] = (1−p)·latency + p·timeout) and adaptive
    per-call timeouts (clamp(3·latency, 6s, 20s)).
  - Per-instance 429 cooldown (jittered) instead of provider-wide skip.
  - JSON-mode fallback when a provider rejects a structured-output schema.
"""

from __future__ import annotations

import os
import logging
import time  # compatibility surface: legacy health tests patch model_registry.time
from pathlib import Path

from dotenv import load_dotenv

from .schemas import ModelClient
from .health import ProviderHealthTracker, normalize_model_id
from .providers import (
    CloudflareNativeVisionClient,
    _build_audio_pipeline,
    _build_premium_audio_pipeline,
    _build_premium_pipeline,
    _build_text_pipeline,
    _build_vision_pipeline,
    _credential_fingerprint,
    _premium_config,
)
from . import routing as _routing
from . import probes as _probes
from . import failover as _failover

# Stable façade exports retained for root and package-internal callers.
DEFAULT_MODEL_TIERS = _routing.DEFAULT_MODEL_TIERS
DEFAULT_WORKER_MODEL_ORDER = _routing.DEFAULT_WORKER_MODEL_ORDER
MODEL_TIERS = _routing.MODEL_TIERS
UNKNOWN_MODEL_TIER = _routing.UNKNOWN_MODEL_TIER
WORKER_MAX_TIER = _routing.WORKER_MAX_TIER
get_model_tier = _routing.get_model_tier
order_failover_chain = _routing.order_failover_chain
route_auxiliary_chain = _routing.route_auxiliary_chain
route_auxiliary_chain_names = _routing.route_auxiliary_chain_names
route_worker_chain = _routing.route_worker_chain
_worker_priority = _routing.worker_priority
AGENTIC_TEXT_ALLOWLIST = _probes.AGENTIC_TEXT_ALLOWLIST
_combo_of = _probes.combo_of
CircuitBreaker = _failover.CircuitBreaker
MODEL_TIMEOUT_FLOOR_SECONDS = _failover.MODEL_TIMEOUT_FLOOR_SECONDS
MODEL_FAILOVER_BUDGET_SECONDS = _failover.MODEL_FAILOVER_BUDGET_SECONDS
MODEL_FAILOVER_MAX_ATTEMPTS = _failover.MODEL_FAILOVER_MAX_ATTEMPTS
VISION_PROVIDER_TIMEOUT_BURST = _failover.VISION_PROVIDER_TIMEOUT_BURST
ROLE_MAX_ATTEMPTS = _failover.ROLE_MAX_ATTEMPTS
PROCESS_RUN_ID = _failover.PROCESS_RUN_ID
PROVIDER_FAILURE_CLASSES = _failover.PROVIDER_FAILURE_CLASSES
_classify_provider_error = _failover.classify_provider_error
_extract_provider_error_metadata = _failover._extract_provider_error_metadata
_extract_text = _failover.extract_text
_estimate_input_tokens = _failover.estimate_input_tokens
_extract_json_payload = _failover.extract_json_payload
_compact_provider_error = _failover._compact_provider_error
_retry_after_seconds = _failover.retry_after_seconds
_hard_timeout = _failover.hard_timeout
_invoke_json_mode = _failover.invoke_json_mode
_compact_messages_for_retry = _failover.compact_messages_for_retry
_as_model_clients = _failover._as_model_clients
invoke_with_failover = _failover.invoke_with_failover

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH)

try:
    from agent_first_browse.logging import get_logger

    logger = get_logger("model_registry")
except ImportError:
    logger = logging.getLogger("model_registry")


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _env_flag(name: str, default: bool = True) -> bool:
    """Read an environment boolean without treating placeholders as enabled."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}




def get_agent_mode() -> str:
    """'premium' | 'free'. AGENT_MODE=auto (default) → premium iff a PREMIUM_API_KEY
    is set; 'premium'/'free' force the respective path."""
    mode = os.getenv("AGENT_MODE", "auto").strip().lower()
    if mode == "premium":
        return "premium"
    if mode == "free":
        return "free"
    return "premium" if os.getenv("PREMIUM_API_KEY", "").strip() else "free"


# ═══════════════════════════════════════════════════════════════════════════════
#  ModelRegistry Singleton
# ═══════════════════════════════════════════════════════════════════════════════


class ModelRegistry:
    """Centralized registry with strictly separated text/vision/audio clients."""

    _instance: "ModelRegistry | None" = None

    def __init__(self):
        # current dual-mode: premium (one trusted paid key) vs free (juggle + gate).
        self.mode = get_agent_mode()
        if self.mode == "premium":
            self._text_pipeline, self._vision_pipeline = _build_premium_pipeline()
            self._audio_pipeline = _build_premium_audio_pipeline()
            self._probed = True  # premium is trusted — never probe or capability-gate
            if not self._text_pipeline:
                logger.warning(
                    "⭐ PREMIUM misconfigured (need PREMIUM_API_KEY + "
                    "PREMIUM_MODEL) — falling back to FREE tier."
                )
                self.mode = "free"
                self._probed = False
        if self.mode == "free":
            self._text_pipeline = _build_text_pipeline()
            self._vision_pipeline = _build_vision_pipeline()
            self._audio_pipeline = _build_audio_pipeline()
            self._probed = False
        self._vision_probed = self.mode == "premium"
        health_path: Path | None = None
        if _env_flag("MODEL_HEALTH_PERSISTENCE", True):
            configured_path = os.getenv(
                "MODEL_HEALTH_PATH", "persistence/model_health.json"
            ).strip()
            if configured_path:
                health_path = Path(configured_path)
                if not health_path.is_absolute():
                    health_path = _PROJECT_ROOT / health_path
        self.health = ProviderHealthTracker(health_path)
        self.health.register_clients(
            self._text_pipeline + self._vision_pipeline + self._audio_pipeline
        )
        # UCRF: Higher tolerance — LLM failovers are expected, not critical failures
        # With 9-model chain, 3-4 timeouts per call is normal before success
        self.breaker = CircuitBreaker(
            window_size=50,
            min_calls=20,  # Need 20+ calls before evaluating (was 10)
            failure_rate_threshold=0.70,  # 70% failure rate to trip (was 50%)
            open_wait_secs=10.0,  # Recover faster (was 30s)
            half_open_probes=3,
        )

        # Validate no overlap
        t = {m.name for m in self._text_pipeline}
        v = {m.name for m in self._vision_pipeline}
        a = {m.name for m in self._audio_pipeline}
        overlap = (t & v) | (t & a) | (v & a)
        if overlap:
            raise RuntimeError(f"Pipeline overlap: {overlap}")

        logger.info(
            "ModelRegistry [%s mode] — TEXT: %d models, VISION: %d models, AUDIO: %d models",
            self.mode,
            len(self._text_pipeline),
            len(self._vision_pipeline),
            len(self._audio_pipeline),
        )

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def get_text_chain(self) -> list[ModelClient]:
        return list(self._text_pipeline)

    def get_text_chain_names(self) -> list[str]:
        return [m.name for m in self._text_pipeline]

    def has_text_models(self) -> bool:
        return len(self._text_pipeline) > 0

    def get_vision_chain(self) -> list[ModelClient]:
        return list(self._vision_pipeline)

    def get_vision_chain_names(self) -> list[str]:
        return [m.name for m in self._vision_pipeline]

    def has_vision_models(self) -> bool:
        return len(self._vision_pipeline) > 0

    def get_audio_chain(self) -> list[ModelClient]:
        return list(self._audio_pipeline)

    def get_audio_chain_names(self) -> list[str]:
        return [m.name for m in self._audio_pipeline]

    def has_audio_models(self) -> bool:
        return len(self._audio_pipeline) > 0

    def get_worker_chain(self) -> list[ModelClient]:
        """current role separation — the WORKER (action decisions, the critical path)
        draws only from the most capable models. Premium → the single premium
        model. Free → tier ≤ WORKER_MAX_TIER, ordered by WORKER_MODEL_ORDER.
        Chronically unreliable instances are omitted so the worker fails fast
        instead of paying their timeout tax on every survey question."""
        return route_worker_chain(self._text_pipeline, self.health, self.mode)

    def get_worker_chain_names(self) -> list[str]:
        return [mc.name for mc in self.get_worker_chain()]

    def get_auxiliary_chain(self) -> list[ModelClient]:
        """Return a provider-prioritized chain for high-volume support calls.

        The worker still uses capability tiers. Planner, PRM, WebDreamer,
        reality reconciliation and the outcome judge instead prefer providers
        intended for high-volume free inference, keeping the critical browser
        decision path isolated.
        """
        return route_auxiliary_chain(self._text_pipeline, self.mode)

    def get_auxiliary_chain_names(self) -> list[str]:
        return route_auxiliary_chain_names(self._text_pipeline, self.health, self.mode)

    def _apply_capability_gate(
        self,
        pipeline: list[ModelClient],
        dead: set,
        incapable: set,
        label: str,
    ) -> list[ModelClient]:
        """Compatibility façade for the probe-owned capability gate."""
        return _probes.apply_capability_gate(pipeline, dead, incapable, label)

    def summary(self) -> str:
        t = " → ".join(self.get_text_chain_names()) or "(none)"
        v = " → ".join(self.get_vision_chain_names()) or "(none)"
        return (
            f"TEXT chain ({len(self._text_pipeline)}): {t}\nVISION chain ({len(self._vision_pipeline)}): {v}"
        )

    # ── Startup probe — prune dead models and seed latency ─────────────────────

    async def probe_and_prune(
        self, timeout: float = 8.0, vision_timeout: float | None = None,
        *, probe_vision: bool = True,
    ) -> None:
        """Probe one instance per (provider, base_model, pipeline) concurrently.

        - 404 / not-found / invalid-key → ALL instances of that combo are
          removed from the chain for this session (no per-step 404 tax).
        - Success → measured latency seeds the EWMA for every sibling instance.
        - Timeout → kept (may be transient) but seeded with a high latency so
          it sorts to the back of its tier.
        - 429 → model exists; kept, no penalty.

        Idempotent: runs once per process. Total wall cost ≈ one round-trip.
        """
        if self._probed:
            return
        self._probed = True
        self._text_pipeline, self._vision_pipeline = await _probes.probe_and_prune(
            self._text_pipeline,
            self._vision_pipeline,
            self.health,
            timeout=timeout,
            vision_timeout=vision_timeout,
            probe_vision=probe_vision,
        )

    async def ensure_vision_capability(
        self, timeout: float = 8.0, vision_timeout: float | None = None,
    ) -> list[ModelClient]:
        """Probe vision once, only when a real visual consult is requested."""
        if getattr(self, "_vision_probed", False):
            return list(self._vision_pipeline)
        self._vision_probed = True
        _text, self._vision_pipeline = await _probes.probe_and_prune(
            [], self._vision_pipeline, self.health,
            timeout=timeout, vision_timeout=vision_timeout or timeout,
            probe_vision=True,
        )
        return list(self._vision_pipeline)
