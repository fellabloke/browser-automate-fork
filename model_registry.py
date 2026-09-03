"""ModelRegistry — Model-Agnostic Provider Layer for Agent First IDE v17.0.

Enforces STRICT separation between Text, Vision, and Audio API pipelines.
No graph node should ever construct an LLM client directly.
All model access goes through this registry.

Reads ALL API keys from .env — NVIDIA NIM, Cloudflare, Google Gemini.

V17.0 — Resilient Reasoning Foundation:
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

import asyncio
import hashlib
import json
import os
import random
import re
import time
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH)

try:
    from app.logger import get_logger

    logger = get_logger("model_registry")
except ImportError:
    logger = logging.getLogger("model_registry")


# ═══════════════════════════════════════════════════════════════════════════════
#  Environment Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_keys(*env_vars: str) -> list[str]:
    """Collect all unique API keys from multiple env vars (comma-separated)."""
    keys: list[str] = []
    for var in env_vars:
        raw = os.getenv(var, "").strip()
        if not raw:
            continue
        for k in raw.replace(";", ",").split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


def _env_flag(name: str, default: bool = True) -> bool:
    """Read an environment boolean without treating placeholders as enabled."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


MODEL_TIMEOUT_FLOOR_SECONDS = _float_env(
    "MODEL_TIMEOUT_FLOOR_SECONDS", 5.0, minimum=1.0
)
# Main worker calls previously had no total budget when the caller used the
# compatibility wrapper's defaults. A provider could therefore walk every key
# and model indefinitely while each individual timeout looked reasonable.
MODEL_FAILOVER_BUDGET_SECONDS = _float_env(
    "MODEL_FAILOVER_BUDGET_SECONDS", 15.0, minimum=1.0
)
MODEL_FAILOVER_MAX_ATTEMPTS = _int_env(
    "MODEL_FAILOVER_MAX_ATTEMPTS", 5, minimum=1
)
MODEL_TIMEOUT_RETRY_COOLDOWN_SECONDS = _float_env(
    "MODEL_TIMEOUT_RETRY_COOLDOWN_SECONDS", 60.0, minimum=1.0
)
MODEL_TIMEOUT_RETRY_COOLDOWN_MAX_SECONDS = _float_env(
    "MODEL_TIMEOUT_RETRY_COOLDOWN_MAX_SECONDS", 60.0, minimum=1.0
)
VISION_PROVIDER_TIMEOUT_BURST = _int_env(
    "VISION_PROVIDER_TIMEOUT_BURST", 1, minimum=1
)
ROLE_MAX_ATTEMPTS = {
    "TEXT_WORKER": _int_env("TEXT_WORKER_MAX_ATTEMPTS", 2, minimum=1),
    "VISION": _int_env("VISION_MAX_ATTEMPTS", 2, minimum=1),
    "CAPTCHA": _int_env("CAPTCHA_MAX_ATTEMPTS", 2, minimum=1),
    "AUXILIARY": _int_env("AUXILIARY_MAX_ATTEMPTS", 1, minimum=1),
}
PROCESS_RUN_ID = os.getenv("RUN_ID", "") or datetime.now().strftime("run_%Y%m%d_%H%M%S")

PROVIDER_FAILURE_CLASSES = (
    "SUCCESS", "TIMEOUT", "RATE_LIMIT", "QUOTA", "HTTP_ERROR",
    "MALFORMED_STRUCTURED_OUTPUT", "EMPTY_STRUCTURED_OUTPUT",
    "SCHEMA_INCOMPATIBILITY", "TRANSPORT_ERROR", "SAFETY_REFUSAL", "UNKNOWN",
)


def _extract_provider_error_metadata(exc: BaseException, response: Any = None) -> dict[str, Any]:
    """Return bounded, secret-free diagnostics shared by every provider branch."""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    response_obj = getattr(exc, "response", None)
    status = status or getattr(response_obj, "status_code", None)
    body = getattr(exc, "body", None) or getattr(exc, "message", None)
    if not body and response_obj is not None:
        body = getattr(response_obj, "text", None)
    if not body:
        body = str(exc) or repr(exc)
    diagnostic = _compact_provider_error(str(body))[:500]
    raw = str(body).encode("utf-8", errors="ignore")
    return {
        "exception_class": type(exc).__name__,
        "http_status": int(status) if str(status).isdigit() else None,
        "response_size": len(raw) if raw else (len(str(response)) if response is not None else 0),
        "response_hash": hashlib.sha256(raw).hexdigest()[:16] if raw else "",
        "diagnostic": diagnostic,
    }


def _classify_provider_error(
    exc: BaseException, *, schema: type | None = None, response: Any = None
) -> tuple[str, dict[str, Any]]:
    """Normalize provider failures before routing or health accounting."""
    metadata = _extract_provider_error_metadata(exc, response)
    status = metadata["http_status"]
    text = metadata["diagnostic"].lower()
    cls = metadata["exception_class"].lower()
    if isinstance(exc, asyncio.TimeoutError) or "timeout" in cls:
        category = "TIMEOUT"
    elif status == 429 or any(x in text for x in ("rate limit", "rate_limit", "resource_exhausted")):
        category = "QUOTA" if any(x in text for x in ("daily", "per day", "quota exceeded", "allocation")) else "RATE_LIMIT"
    elif (status is not None and status >= 400) or re.search(
        r"(?:error\s+code|http(?:\s+status)?|status)\s*[:=]?\s*[45]\d{2}", text,
    ):
        category = "SCHEMA_INCOMPATIBILITY" if schema is not None and any(x in text for x in ("schema", "response_format", "additionalproperties")) else "HTTP_ERROR"
    elif schema is not None and (isinstance(exc, json.JSONDecodeError) or any(
        x in text for x in ("expecting value", "json_invalid", "validation error", "parsed field")
    )):
        category = "EMPTY_STRUCTURED_OUTPUT" if isinstance(exc, json.JSONDecodeError) and "column 1" in text else "MALFORMED_STRUCTURED_OUTPUT"
    elif any(x in text for x in ("safety refusal", "content policy", "refused to answer")):
        category = "SAFETY_REFUSAL"
    elif isinstance(exc, (ConnectionError, OSError)) or any(x in cls for x in ("transport", "connection")):
        category = "TRANSPORT_ERROR"
    else:
        category = "UNKNOWN"
    metadata["classification"] = category
    return category, metadata


def _credential_fingerprint(secret: str) -> str:
    """Stable, non-reversible key identity used by the health cache.

    Raw API keys must never be persisted or logged. The short digest is only
    used to keep the same project's health attached when key ordering changes.
    """
    return hashlib.sha256(str(secret).encode("utf-8")).hexdigest()[:16]


def _gemini_project_limit(instance_name: str, metric: str) -> int:
    """Resolve an optional per-project limit list by the current key slot."""
    singular = f"GEMINI_PROJECT_{metric.upper()}_LIMIT"
    plural = singular + "S"
    values = [
        value.strip()
        for value in re.split(r"[,;]", os.getenv(plural, ""))
    ]
    try:
        index = int(instance_name.rsplit(":", 1)[-1])
        if index < len(values) and values[index]:
            return max(0, int(values[index]))
    except (TypeError, ValueError):
        pass
    return _int_env(singular, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Quality Tiers — V17.0 model-first failover
#
#  The failover chain is sorted by (role priority, tier, expected_cost). Worker
#  chains assign explicit provider/model priorities; auxiliary chains prioritize
#  roomier providers. Lower tier = higher quality/capability eligibility.
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL_TIERS: dict[str, int] = {
    # Tier 0 — proven structured-output models. Exact provider/model worker
    # ordering is handled separately so NVIDIA's GPT endpoint can be demoted
    # without changing GPT OSS consensus semantics globally.
    "gemini-3.5-flash-lite": 0,
    "gpt-oss-120b": 0,
    # Tier 1 — capable worker fallbacks. Provider/model role priority determines
    # their exact order; the tier gate merely admits them to the worker pool.
    "nemotron-3.5-lightning-30b-a3b": 1,
    "llama-3.3-70b-instruct-fp8-fast": 1,
    "gemma-4-31b-it": 1,
    "gemma-4-32b-it": 1,
    "gemma-4-26b-a4b-it": 1,
    # ── Vision-only models (NEVER used by the text chain — tiering them here is
    #    safe and does not touch text ordering). Without this, the proven-fast
    #    fast vision models must stay ahead of slower fallbacks. ──
    "llama-4-scout-17b-16e-instruct": 0,
    "gemini-3.5-flash": 0,  # configured Google vision primary; keep ahead of NVIDIA fallbacks
    "llama-3.2-11b-vision-instruct": 1,  # NVIDIA — fast, lightweight (~0.6s)
    "llama-3.1-nemotron-nano-vl-8b-v1": 1,  # NVIDIA — fast nano VL (~0.7s)
    "llama-4-maverick-17b-128e-instruct": 1,  # NVIDIA — strong, 1M ctx native vision
    "llama-3.2-90b-vision-instruct": 2,  # NVIDIA — strong 90B fallback
}
UNKNOWN_MODEL_TIER = 3


def normalize_model_id(model_id: str) -> str:
    """'openai/gpt-oss-120b' → 'gpt-oss-120b'; 'google/gemma-4-31b-it' → 'gemma-4-31b-it'."""
    return model_id.rsplit("/", 1)[-1].strip().lower()


def _load_tier_overrides() -> dict[str, int]:
    """Parse MODEL_TIER_OVERRIDE='gpt-oss-120b=0,some-model=2' from .env."""
    overrides: dict[str, int] = {}
    raw = os.getenv("MODEL_TIER_OVERRIDE", "").strip()
    for part in raw.split(","):
        if "=" in part:
            mid, _, rank = part.partition("=")
            try:
                overrides[normalize_model_id(mid)] = int(rank)
            except ValueError:
                pass
    return overrides


MODEL_TIERS: dict[str, int] = {**DEFAULT_MODEL_TIERS, **_load_tier_overrides()}


def get_model_tier(instance_name: str) -> int:
    """Quality tier for an instance name like 'nvidia-text:openai/gpt-oss-120b:0'."""
    base = ProviderHealthTracker._base_model_name(instance_name)
    return MODEL_TIERS.get(normalize_model_id(base), UNKNOWN_MODEL_TIER)


# ═══════════════════════════════════════════════════════════════════════════════
#  V24 — Agentic Capability Gate + dual-mode (free / premium). See STRATEGY.md.
# ═══════════════════════════════════════════════════════════════════════════════

# Normalized base model names PROVEN (live probe, 2026-06) to do agentic
# structured output. Shipped default + SAFETY FLOOR for the capability gate
# (if gating would empty a pipeline, these alive models are restored).
AGENTIC_TEXT_ALLOWLIST: set[str] = {
    "gemini-3.5-flash-lite",  # Google — fast and reliable structured output
    "gpt-oss-120b",  # NVIDIA fallback
    "nemotron-3.5-lightning-30b-a3b",  # NVIDIA — agentic replacement for OSS 20B
    "llama-3.3-70b-instruct-fp8-fast",  # Cloudflare — JSON/function-call fallback
    "gemma-4-31b-it",  # via Gemini (fails on NVIDIA → probe drops that combo)
    "gemma-4-32b-it",
    "gemma-4-26b-a4b-it",  # Cloudflare — efficient structured-output fallback
    "llama-3.3-70b-instruct",  # NVIDIA — works, slower
    "llama-3.3-nemotron-super-49b-v1.5",  # NVIDIA — works, medium
}

# Role separation: the worker (action decisions) only uses models at or below
# this tier. Auxiliary calls (planner, PRM, judge) use the full chain.
WORKER_MAX_TIER = 1

DEFAULT_WORKER_MODEL_ORDER = (
    "google:gemini-3.5-flash-lite,"
    "nvidia:nemotron-3.5-lightning-30b-a3b,"
    "cloudflare:llama-3.3-70b-instruct-fp8-fast,"
    "nvidia:gpt-oss-120b,"
)


def _worker_priority(mc: "ModelClient") -> int:
    """Resolve explicit provider+model priority for critical worker calls."""
    requested = [
        entry.strip().lower()
        for entry in os.getenv("WORKER_MODEL_ORDER", DEFAULT_WORKER_MODEL_ORDER).split(",")
        if entry.strip()
    ]
    base_model = normalize_model_id(ProviderHealthTracker._base_model_name(mc.name))
    provider = mc.provider.lower()
    for priority, selector in enumerate(requested):
        selected_provider, separator, selected_model = selector.partition(":")
        if not separator:
            continue
        if selected_provider == provider and normalize_model_id(selected_model) == base_model:
            return priority
    return len(requested)


def get_agent_mode() -> str:
    """'premium' | 'free'. AGENT_MODE=auto (default) → premium iff a PREMIUM_API_KEY
    is set; 'premium'/'free' force the respective path."""
    mode = os.getenv("AGENT_MODE", "auto").strip().lower()
    if mode == "premium":
        return "premium"
    if mode == "free":
        return "free"
    return "premium" if os.getenv("PREMIUM_API_KEY", "").strip() else "free"


def _premium_config() -> dict:
    """Premium single-key/model config — one paid key that does text + vision."""
    return {
        "keys": _collect_keys("PREMIUM_API_KEY", "PREMIUM_API_KEYS"),
        "model": os.getenv("PREMIUM_MODEL", "").strip(),
        "vision_model": os.getenv("PREMIUM_VISION_MODEL", "").strip(),
        "base_url": os.getenv("PREMIUM_BASE_URL", "https://api.openai.com/v1").strip(),
        "provider": os.getenv("PREMIUM_PROVIDER", "openai").strip().lower(),
        "timeout": int(os.getenv("PREMIUM_TIMEOUT", "60")),
    }


def _combo_of(mc: "ModelClient") -> tuple[str, str, str]:
    """(provider, normalized_base_model, pipeline) — the probe/gate identity key."""
    return (mc.provider, normalize_model_id(ProviderHealthTracker._base_model_name(mc.name)), mc.pipeline)


# ═══════════════════════════════════════════════════════════════════════════════
#  Provider Health Tracker
# ═══════════════════════════════════════════════════════════════════════════════


class ProviderHealthTracker:
    """Per-model exponential backoff quarantine with V15.0/V15.1 improvements.

    V15.0 F6:
      - Quarantine cap at 16s (was uncapped up to 120s)
      - Cold-start reset: if model idle 5+ min, streak resets to 0
      - Schema blacklist: models that fail structured output permanently

    V15.1 Patch A+C:
      - Base-model-name blacklist applies to all sibling instances of a model
        also blocks ':1' and ':2' (same underlying model, different API key)
      - Per-schema blacklist: 'ChecklistEvaluation' vs 'CandidateSet' tracked
        separately so a model that fails on one schema can still serve others
    """

    # EWMA smoothing factor for latency / failure-probability estimates
    EWMA_ALPHA = 0.3
    # Assumed latency (s) for an instance we have never measured
    DEFAULT_LATENCY = 8.0

    def __init__(self, persistence_path: str | Path | None = None):
        self._health: dict[str, dict] = {}
        self._schema_blacklist: dict[str, set[str]] = {}  # "provider|base_model" → {schema_names}
        self._aliases: dict[str, str] = {}
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._loaded_mtime_ns = 0
        # A network/model timeout is normally shared by sibling API keys for the
        # same provider/model.  Keep this separate from 429 cooldowns, which are
        # deliberately per-key because quota can differ between keys/projects.
        self._timeout_cooldowns: dict[str, dict[str, float]] = {}
        # Process-scoped role affinity; successful primaries stay sticky for a run.
        self._role_affinity: dict[str, str] = {}
        self._load()

    @staticmethod
    def _default_state() -> dict[str, Any]:
        return {
            "consecutive_failures": 0,
            "quarantine_until": 0.0,
            "total_calls": 0,
            "total_failures": 0,
            "last_call_time": 0.0,
            "last_success_wall": 0.0,
            "last_failure_wall": 0.0,
            "latency_ewma": None,
            "p_fail": 0.0,
            # Wall-clock deadlines survive process restarts.
            "cooldown_until": 0.0,
            "daily_exhausted_until": 0.0,
            "usage_day": "",
            "daily_requests": 0,
            "malformed_output_count": 0,
            "structured_repair_successes": 0,
            "structured_repair_failures": 0,
            "last_failure_class": "",
            # [unix_timestamp, estimated_input_tokens]. Only one minute retained.
            "request_events": [],
        }

    def _load(self) -> None:
        if self._persistence_path is None or not self._persistence_path.is_file():
            return
        try:
            payload = json.loads(self._persistence_path.read_text(encoding="utf-8"))
            cache_version = int(payload.get("version", 1) or 1)
            states = payload.get("states", {})
            if isinstance(states, dict):
                for identity, saved in states.items():
                    if not isinstance(saved, dict):
                        continue
                    state = self._default_state()
                    state.update({key: value for key, value in saved.items() if key in state})
                    state["request_events"] = [
                        event for event in state.get("request_events", [])
                        if isinstance(event, list) and len(event) == 2
                    ][-500:]
                    if cache_version < 3:
                        # V2 counted 429 quota responses and 413 request-size
                        # errors as endpoint unreliability. Those counters can
                        # permanently bench a valid model after a busy run.
                        # Keep latency/quota ledgers, but relearn reliability
                        # under the corrected classification rules.
                        state.update({
                            "consecutive_failures": 0,
                            "total_failures": 0,
                            "last_failure_wall": 0.0,
                            "p_fail": 0.0,
                        })
                    self._health[str(identity)] = state
            blacklists = payload.get("schema_blacklist", {})
            if isinstance(blacklists, dict):
                self._schema_blacklist = {
                    str(key): {str(item) for item in value}
                    for key, value in blacklists.items()
                    if isinstance(value, list)
                }
            try:
                self._loaded_mtime_ns = self._persistence_path.stat().st_mtime_ns
            except OSError:
                self._loaded_mtime_ns = 0
        except Exception as exc:  # noqa: BLE001 - a stale cache must never stop the agent
            logger.warning("Ignoring unreadable model health cache: %s", exc)

    def _refresh_if_changed(self) -> None:
        """Pick up an external usage reset while the agent process is alive."""
        if self._persistence_path is None or not self._persistence_path.is_file():
            return
        try:
            mtime_ns = self._persistence_path.stat().st_mtime_ns
        except OSError:
            return
        if mtime_ns != self._loaded_mtime_ns:
            self._load()
            logger.info("♻️ Reloaded model health cache changed by another process")

    def _save(self) -> None:
        if self._persistence_path is None:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 3,
                "updated_at": time.time(),
                "states": self._health,
                "schema_blacklist": {
                    key: sorted(value) for key, value in self._schema_blacklist.items()
                },
            }
            temporary = self._persistence_path.with_suffix(
                self._persistence_path.suffix + ".tmp"
            )
            temporary.write_text(
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self._persistence_path)
            self._loaded_mtime_ns = self._persistence_path.stat().st_mtime_ns
        except Exception as exc:  # noqa: BLE001 - health persistence is best effort
            logger.debug("Could not persist model health cache: %s", exc)

    def register_clients(self, clients: list["ModelClient"]) -> None:
        """Attach runtime index names to stable anonymous credential identities."""
        for client in clients:
            if client.credential_id:
                identity_parts = [client.provider.lower()]
                # Gemini quota is project+model scoped. Vision and audio calls
                # using the same model/key must consume one shared ledger.
                if client.provider.lower() != "google":
                    identity_parts.append(client.pipeline.lower())
                identity_parts.extend((
                    normalize_model_id(self._base_model_name(client.name)),
                    client.credential_id,
                ))
                identity = "|".join(identity_parts)
            else:
                identity = client.name
            self._aliases[client.name] = identity
            if identity not in self._health:
                previous = self._health.pop(client.name, None)
                self._health[identity] = previous or self._default_state()

    def _identity(self, name: str) -> str:
        return self._aliases.get(name, name)

    @staticmethod
    def _base_model_name(instance_name: str) -> str:
        """Extract base model from instance name.

        'nvidia-text:openai/gpt-oss-120b:0'  → 'openai/gpt-oss-120b'
        'nvidia-text:z-ai/glm-5.1:1'  → 'z-ai/glm-5.1'
        'gemini-text:gemma-4-31b-it:2' → 'gemma-4-31b-it'
        'some-model'                   → 'some-model' (fallback)
        """
        parts = instance_name.split(":")
        if len(parts) >= 3:
            # provider:model_path:key_index → extract model_path
            return ":".join(parts[1:-1])
        return instance_name

    @staticmethod
    def _blacklist_key(instance_name: str) -> str:
        """Blacklist key scoped to (provider, base_model).

        V17.0: schema support is a property of the provider+model combo, not
        the model name globally — gpt-oss may reject a schema on Groq while
        serving it fine on NVIDIA. Never bench a good model globally.
        """
        provider = instance_name.split(":", 1)[0]
        base = ProviderHealthTracker._base_model_name(instance_name)
        return f"{provider}|{base}"

    @staticmethod
    def _timeout_scope(instance_name: str) -> str:
        """Return the persistent timeout identity.

        Timeout quarantine follows the individual credential/project. A
        stalled key must not suppress sibling projects with independent
        capacity.
        """
        return instance_name

    @staticmethod
    def _family_scope(instance_name: str) -> str:
        provider_pipeline = instance_name.split(":", 1)[0]
        base = ProviderHealthTracker._base_model_name(instance_name)
        return f"{provider_pipeline}|{base}"

    def _get(self, name: str) -> dict:
        identity = self._identity(name)
        if identity not in self._health:
            self._health[identity] = self._default_state()
        return self._health[identity]

    # ── V17.0: EWMA + cooldown API ─────────────────────────────────────────

    def observe_latency(self, name: str, seconds: float) -> None:
        """Fold an observed call latency into the EWMA estimate."""
        s = self._get(name)
        if s["latency_ewma"] is None:
            s["latency_ewma"] = seconds
        else:
            s["latency_ewma"] = self.EWMA_ALPHA * seconds + (1 - self.EWMA_ALPHA) * s["latency_ewma"]

    def seed_latency(self, name: str, seconds: float) -> None:
        """Seed the latency estimate (from the startup probe) without overriding live data."""
        s = self._get(name)
        if s["latency_ewma"] is None:
            s["latency_ewma"] = seconds
            self._save()

    def _update_p_fail(self, name: str, failed: bool) -> None:
        s = self._get(name)
        s["p_fail"] = self.EWMA_ALPHA * (1.0 if failed else 0.0) + (1 - self.EWMA_ALPHA) * s["p_fail"]

    def start_cooldown(self, name: str, seconds: float) -> None:
        """Put one instance (provider+key+model) on a rate-limit cooldown.

        Per-instance, NOT per-provider: limits may be project/credential scoped,
        so a 429 on key 0 says nothing about key 1. The same model on the next
        independent key is tried immediately.
        """
        s = self._get(name)
        s["cooldown_until"] = time.time() + seconds
        self._save()
        logger.info("⏸️ '%s' on 429 cooldown for %.0fs (same model continues on other keys)", name, seconds)

    def in_cooldown(self, name: str, *, critical: bool = False) -> bool:
        self._refresh_if_changed()
        state = self._get(name)
        now = time.time()
        if now < float(state.get("cooldown_until", 0.0) or 0.0):
            return True
        return self.in_quota_guard(name, critical=critical)

    def in_quota_guard(self, name: str, *, critical: bool = False) -> bool:
        """True when another request could cross its role-specific free limit.

        High-volume auxiliary work stops earlier, leaving a quota reserve for
        critical worker decisions.  Critical calls may consume that reserve but
        retain a small final safety margin so a local estimate never walks right
        into the provider's hard limit.
        """
        state = self._get(name)
        now = time.time()
        # RPD is a calendar-day quota in Google's timezone, not a rolling
        # 24-hour window.  Roll the local ledger before checking any parking
        # state so a key becomes eligible on the first scheduling pass after
        # Google's reset, even when the process stayed alive across midnight.
        self._roll_daily_usage(name, now)
        if now < float(state.get("daily_exhausted_until", 0.0) or 0.0):
            return True
        if not name.startswith("gemini-"):
            return False
        usage = self.quota_snapshot(name)
        fallback = _float_env(
            "GEMINI_USAGE_SOFT_LIMIT_PERCENT", 90.0, minimum=1.0
        )
        soft = _float_env(
            (
                "GEMINI_WORKER_USAGE_LIMIT_PERCENT"
                if critical
                else "GEMINI_AUXILIARY_USAGE_LIMIT_PERCENT"
            ),
            98.0 if critical else min(80.0, fallback),
            minimum=1.0,
        ) / 100.0
        limits = {
            metric: _gemini_project_limit(name, metric)
            for metric in ("rpm", "tpm", "rpd")
        }
        return any(
            limit > 0 and usage[metric] >= max(1, int(limit * soft))
            for metric, limit in limits.items()
        )

    @staticmethod
    def _pacific_day_key(now: float | None = None) -> str:
        current = datetime.fromtimestamp(now or time.time(), ZoneInfo("America/Los_Angeles"))
        return current.date().isoformat()

    @staticmethod
    def _next_pacific_midnight(now: float | None = None) -> float:
        current = datetime.fromtimestamp(now or time.time(), ZoneInfo("America/Los_Angeles"))
        tomorrow = (current + timedelta(days=1)).date()
        return datetime(
            tomorrow.year, tomorrow.month, tomorrow.day,
            tzinfo=ZoneInfo("America/Los_Angeles"),
        ).timestamp()

    @staticmethod
    def _quota_day_key(name: str, now: float | None = None) -> str:
        """Use the provider's calendar for daily allocation bookkeeping."""
        if name.lower().startswith("cloudflare|"):
            return datetime.fromtimestamp(now or time.time(), ZoneInfo("UTC")).date().isoformat()
        return ProviderHealthTracker._pacific_day_key(now)

    @staticmethod
    def _next_quota_midnight(name: str, now: float | None = None) -> float:
        if name.lower().startswith("cloudflare|"):
            current = datetime.fromtimestamp(now or time.time(), ZoneInfo("UTC"))
            tomorrow = (current + timedelta(days=1)).date()
            return datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                            tzinfo=ZoneInfo("UTC")).timestamp()
        return ProviderHealthTracker._next_pacific_midnight(now)

    def _roll_daily_usage(self, name: str, now: float | None = None) -> bool:
        """Reset persisted usage when the provider's calendar day changes.

        Gemini resets at midnight Pacific while Cloudflare's daily allocation
        uses UTC.  This is kept
        as an explicit operation rather than embedding the reset in one
        caller, because quota eligibility, snapshots, and request accounting
        can all be the first path exercised after a process crosses midnight.
        Returns whether this key's ledger changed.
        """
        now = time.time() if now is None else now
        state = self._get(name)
        day_key = self._quota_day_key(name, now)
        changed = False
        if state.get("usage_day") != day_key:
            previous_day = state.get("usage_day") or "unknown"
            state["usage_day"] = day_key
            state["daily_requests"] = 0
            state["daily_exhausted_until"] = 0.0
            changed = True
            if name.startswith("gemini-"):
                logger.info(
                    "♻️ Gemini daily quota ledger reset for %s: %s → %s (midnight Pacific)",
                    name,
                    previous_day,
                    day_key,
                )
        elif float(state.get("daily_exhausted_until", 0.0) or 0.0) <= now and state.get("daily_exhausted_until"):
            # Clear an old persisted hard-limit marker once its scheduled
            # reset has passed, even if the day key was already normalized.
            state["daily_exhausted_until"] = 0.0
            changed = True
        if changed:
            self._save()
        return changed

    def _prune_request_events(self, state: dict, now: float) -> None:
        cutoff = now - 60.0
        state["request_events"] = [
            [float(event[0]), int(event[1])]
            for event in state.get("request_events", [])
            if isinstance(event, (list, tuple)) and len(event) == 2
            and float(event[0]) >= cutoff
        ][-500:]

    def record_attempt(self, name: str, estimated_input_tokens: int = 0) -> None:
        """Record local usage before an API request for quota-aware scheduling."""
        now = time.time()
        state = self._get(name)
        self._roll_daily_usage(name, now)
        self._prune_request_events(state, now)
        state["daily_requests"] = int(state.get("daily_requests", 0) or 0) + 1
        state["request_events"].append([now, max(0, int(estimated_input_tokens))])
        self._save()

    def quota_snapshot(self, name: str) -> dict[str, int]:
        """Return locally observed RPM/TPM/RPD for one anonymous project key."""
        self._refresh_if_changed()
        now = time.time()
        state = self._get(name)
        self._roll_daily_usage(name, now)
        self._prune_request_events(state, now)
        minute = [event for event in state["request_events"] if event[0] >= now - 60.0]
        return {
            "rpm": len(minute),
            "tpm": sum(event[1] for event in minute),
            "rpd": int(state.get("daily_requests", 0) or 0),
        }

    def reset_usage_limits(
        self, providers: tuple[str, ...] = ("google", "cloudflare")
    ) -> int:
        """Clear local usage/cooldown parking for selected providers immediately.

        Dashboard quota resets should make a live process eligible again
        without erasing useful latency or reliability diagnostics.
        """
        prefixes = tuple(f"{provider.lower()}|" for provider in providers)
        reset = 0
        for identity, state in self._health.items():
            if identity.lower().startswith(prefixes):
                day = self._quota_day_key(identity)
                state["usage_day"] = day
                state["daily_requests"] = 0
                state["request_events"] = []
                state["daily_exhausted_until"] = 0.0
                state["cooldown_until"] = 0.0
                reset += 1
        self._save()
        logger.info(
            "♻️ Reset usage limits for %d %s ledger(s)",
            reset,
            ", ".join(providers),
        )
        return reset

    def clear_timeout_parking(self) -> int:
        """Clear persisted timeout parking after a cooldown-policy change."""
        cleared = 0
        for state in self._health.values():
            if float(state.get("cooldown_until", 0.0) or 0.0) > 0:
                state["cooldown_until"] = 0.0
                cleared += 1
        self._timeout_cooldowns.clear()
        self._save()
        return cleared

    def quota_penalty(self, name: str) -> float:
        """Scheduling cost for estimated project usage; lower means more headroom.

        Gemini API keys cannot query remaining project quota directly. This
        local ledger is therefore conservative. Limits copied from AI Studio
        can be supplied in the environment; zero means unknown.
        """
        if not name.startswith("gemini-"):
            return 0.0
        usage = self.quota_snapshot(name)
        ratios: list[float] = []
        for metric in ("rpm", "tpm", "rpd"):
            limit = _gemini_project_limit(name, metric)
            if limit > 0:
                ratios.append(usage[metric] / limit)
        configured_pressure = max(ratios, default=0.0) * 20.0
        fair_share = usage["rpm"] * 1.5 + usage["rpd"] * 0.002
        return configured_pressure + fair_share

    def record_quota_failure(self, name: str, error: str, cooldown: float) -> None:
        """Persist a per-project rate-limit result, including daily exhaustion."""
        # A quota response says that this credential cannot serve *now*; it does
        # not say that the provider/model is unreliable.  Counting it toward
        # chronic-failure benching permanently removed otherwise valid fallbacks
        # after a busy run.
        self.record_failure(
            name, is_rate_limit=True, error_msg=error,
            reliability_failure=False,
            failure_class="QUOTA",
        )
        state = self._get(name)
        lowered = error.lower()
        daily_markers = (
            "requests per day", "request per day", "perday", "per day",
            "rpd", "daily quota", "daily limit", "daily free allocation",
            "code 4006",
        )
        if any(marker in lowered for marker in daily_markers):
            state["daily_exhausted_until"] = self._next_quota_midnight(name)
            reset_zone = "UTC" if name.lower().startswith("cloudflare") else "Pacific"
            logger.warning(
                "📅 '%s' provider reports daily allocation exhausted; parked until the next %s reset",
                name, reset_zone,
            )
        else:
            state["cooldown_until"] = time.time() + cooldown
        self._save()

    def record_timeout(
        self,
        name: str,
        seconds: float,
        sibling_threshold: int = 2,
    ) -> bool:
        """Track repeated timeouts for one credential/project.

        ``sibling_threshold`` is retained for compatibility and controls
        repeated timeouts of this same key; it never suppresses sibling keys.
        """
        scope = self._timeout_scope(name)
        now = time.monotonic()
        state = self._timeout_cooldowns.setdefault(
            scope, {"strikes": 0.0, "last_timeout": 0.0, "until": 0.0}
        )
        if now - state["last_timeout"] > max(seconds, 1.0):
            state["strikes"] = 0.0
        state["strikes"] += 1.0
        state["last_timeout"] = now
        if state["strikes"] >= max(1, sibling_threshold):
            state["until"] = now + seconds
            logger.info(
                "🐢 '%s' timed out %.0f times; cooling this key for %.0fs",
                scope,
                state["strikes"],
                seconds,
            )
            return True
        return False

    def start_timeout_backoff(self, name: str) -> float:
        """Persistently park an instance after repeated consecutive timeouts.

        Short request-local family cooldowns avoid walking sibling keys once.
        This backoff prevents a key/model that has hung hundreds of times from
        re-entering every later worker call during a long autonomous run.
        """
        state = self._get(name)
        failures = int(state.get("consecutive_failures", 0) or 0)
        if failures < 2:
            return 0.0
        multiplier = 2 ** min(max(0, failures - 2), 4)
        seconds = min(
            MODEL_TIMEOUT_RETRY_COOLDOWN_MAX_SECONDS,
            MODEL_TIMEOUT_RETRY_COOLDOWN_SECONDS * multiplier,
        )
        state["cooldown_until"] = max(
            float(state.get("cooldown_until", 0.0) or 0.0),
            time.time() + seconds,
        )
        self._save()
        logger.warning(
            "🐢 '%s' has %d consecutive timeouts; parked for %.0fs",
            name, failures, seconds,
        )
        return seconds

    def clear_timeout_failures(self, name: str) -> None:
        """A sibling succeeded, so the provider/model timeout streak is over."""
        self._timeout_cooldowns.pop(self._timeout_scope(name), None)

    def preferred_for_role(self, role: str, names: set[str]) -> str | None:
        preferred = self._role_affinity.get(str(role).upper())
        return preferred if preferred in names else None

    def set_preferred_for_role(self, role: str, name: str) -> None:
        self._role_affinity[str(role).upper()] = name

    def clear_preferred_for_role(self, role: str, name: str | None = None) -> None:
        key = str(role).upper()
        if name is None or self._role_affinity.get(key) == name:
            self._role_affinity.pop(key, None)

    def in_timeout_cooldown(self, name: str) -> bool:
        scope = self._timeout_scope(name)
        state = self._timeout_cooldowns.get(scope)
        if state is None or state["until"] <= 0:
            return False
        if state["until"] <= time.monotonic():
            self._timeout_cooldowns.pop(scope, None)
            return False
        return True

    def failure_rate(self, name: str) -> float:
        state = self._get(name)
        total = int(state.get("total_calls", 0) or 0)
        return (
            int(state.get("total_failures", 0) or 0) / total
            if total > 0 else 0.0
        )

    def is_chronically_unreliable(self, name: str) -> bool:
        """Bench instances whose persisted record predicts near-certain failure.

        A later success rehabilitates the instance immediately. A timed retry
        also makes it eligible after a short pause, so an agent that runs for
        hours can recover without requiring a restart or startup probe.
        """
        state = self._get(name)
        minimum_calls = _int_env("MODEL_UNRELIABLE_MIN_CALLS", 10, minimum=2)
        threshold = _float_env("MODEL_UNRELIABLE_FAILURE_RATE", 0.90, minimum=0.5)
        consecutive_limit = _int_env("MODEL_UNRELIABLE_CONSECUTIVE_FAILURES", 5, minimum=2)
        total = int(state.get("total_calls", 0) or 0)
        last_success = float(state.get("last_success_wall", 0.0) or 0.0)
        last_failure = float(state.get("last_failure_wall", 0.0) or 0.0)
        recovered = bool(last_success and last_success > last_failure)
        retry_after = _float_env(
            "MODEL_UNRELIABLE_RETRY_SECONDS", 60.0, minimum=10.0
        )
        retry_due = bool(last_failure and time.time() - last_failure >= retry_after)
        return bool(
            not recovered and not retry_due
            and (
                int(state.get("consecutive_failures", 0) or 0) >= consecutive_limit
                or (total >= minimum_calls and self.failure_rate(name) >= threshold)
            )
        )

    def probe_cache_fresh(self, name: str) -> bool:
        state = self._get(name)
        ttl = _float_env("MODEL_PROBE_CACHE_SECONDS", 21600.0, minimum=60.0)
        last_success = float(state.get("last_success_wall", 0.0) or 0.0)
        return bool(
            last_success
            and time.time() - last_success <= ttl
            and not self.is_chronically_unreliable(name)
            and float(state.get("p_fail", 0.0) or 0.0) < 0.5
        )

    def expected_cost(self, name: str, default_timeout: float = 20.0) -> float:
        """E[time-to-answer] = (1−p_fail)·latency + p_fail·timeout.

        Sorting instances ascending by this value minimizes expected
        time-to-first-success (greedy ordered-search optimality).
        """
        s = self._get(name)
        lat = s["latency_ewma"] if s["latency_ewma"] is not None else self.DEFAULT_LATENCY
        p = s["p_fail"]
        return (1.0 - p) * lat + p * default_timeout

    def adaptive_timeout(self, name: str, k: float = 3.0, floor: float = 6.0, cap: float = 20.0) -> float:
        """Per-instance timeout: clamp(k·latency_ewma, floor, cap).

        Unknown models get the full cap; a chronically slow model is
        abandoned at the cap instead of a fixed 30s.
        """
        s = self._get(name)
        if s["latency_ewma"] is None:
            return cap
        return max(floor, min(cap, k * s["latency_ewma"]))

    def is_available(self, name: str, schema_name: str = "") -> bool:
        # V17.0: Check schema blacklist by (provider, base_model)
        if schema_name and self.is_schema_blacklisted(name, schema_name):
            return False

        s = self._get(name)
        # V15.0: Cold-start reset — if model idle 5+ min, treat as fresh
        if (
            not self.is_chronically_unreliable(name)
            and s["last_call_time"] > 0
            and time.time() - s["last_call_time"] > 300
        ):
            s["consecutive_failures"] = 0
            s["quarantine_until"] = 0.0

        return (
            not self.is_chronically_unreliable(name)
            and time.time() >= s["quarantine_until"]
            and not self.in_cooldown(name)
        )

    def record_success(self, name: str, latency: float | None = None) -> None:
        s = self._get(name)
        s["consecutive_failures"] = 0
        s["quarantine_until"] = 0.0
        s["cooldown_until"] = 0.0
        s["total_calls"] += 1
        s["last_call_time"] = time.time()
        s["last_success_wall"] = s["last_call_time"]
        self._update_p_fail(name, failed=False)
        if latency is not None:
            self.observe_latency(name, latency)
        self.clear_timeout_failures(name)
        self._save()

    def record_malformed_output(self, name: str, failure_class: str) -> None:
        """Record bad structured content without quarantining a responsive model."""
        state = self._get(name)
        state["total_calls"] += 1
        state["last_call_time"] = time.time()
        state["malformed_output_count"] = int(state.get("malformed_output_count", 0) or 0) + 1
        state["last_failure_class"] = failure_class
        threshold = _int_env("MODEL_MALFORMED_OUTPUT_PENALTY_THRESHOLD", 3, minimum=2)
        if state["malformed_output_count"] >= threshold:
            state["consecutive_failures"] += 1
            state["total_failures"] += 1
            state["last_failure_wall"] = state["last_call_time"]
            self._update_p_fail(name, failed=True)
        self._save()

    def record_structured_repair(self, name: str, success: bool) -> None:
        state = self._get(name)
        key = "structured_repair_successes" if success else "structured_repair_failures"
        state[key] = int(state.get(key, 0) or 0) + 1
        state["last_failure_class"] = "" if success else "MALFORMED_STRUCTURED_OUTPUT"
        self._save()

    def record_failure(
        self,
        name: str,
        is_rate_limit: bool = False,
        error_msg: str = "",
        latency: float | None = None,
        reliability_failure: bool = True,
        failure_class: str = "",
    ) -> None:
        s = self._get(name)
        if failure_class:
            s["last_failure_class"] = failure_class
        s["total_calls"] += 1
        s["last_call_time"] = time.time()
        s["last_failure_class"] = ""
        if reliability_failure:
            s["consecutive_failures"] += 1
            s["total_failures"] += 1
            s["last_failure_wall"] = s["last_call_time"]
            self._update_p_fail(name, failed=True)
        if latency is not None:
            # A timeout means "at least this slow" — fold it into the estimate
            self.observe_latency(name, latency)

        # V17.0: Schema blacklist keyed by (provider, base_model) — a model that
        # rejects a schema on one provider may serve it fine on another.
        if error_msg and ("400" in error_msg or "Bad Request" in error_msg):
            if "additionalProperties" in error_msg or "schema" in error_msg.lower():
                bl_key = self._blacklist_key(name)
                if bl_key not in self._schema_blacklist:
                    self._schema_blacklist[bl_key] = set()

                # Extract specific schema name from error message
                # Error format: "response_format: 'ChecklistEvaluation': /properties/..."
                schema_match = re.search(r"response_format:\s*'(\w+)'", error_msg)
                if not schema_match:
                    # Fallback: look for CamelCase class names (Pydantic schema pattern)
                    schema_match = re.search(r"'([A-Z]\w{5,})'", error_msg)
                specific_schema = schema_match.group(1) if schema_match else None

                if specific_schema:
                    self._schema_blacklist[bl_key].add(specific_schema)
                    logger.warning(
                        "🚫 '%s' blacklisted for schema '%s' — will use JSON-mode fallback",
                        bl_key,
                        specific_schema,
                    )
                else:
                    # Catch-all: blacklist for all structured output
                    self._schema_blacklist[bl_key].add("__structured_output__")
                    logger.warning(
                        "🚫 '%s' blacklisted for ALL structured output — JSON-mode fallback",
                        bl_key,
                    )
        self._save()

    def is_schema_blacklisted(self, name: str, schema_name: str = "__structured_output__") -> bool:
        """Check if a (provider, model) combo is blacklisted for a specific schema."""
        bl_key = self._blacklist_key(name)
        if bl_key not in self._schema_blacklist:
            return False
        bl = self._schema_blacklist[bl_key]
        return schema_name in bl or "__structured_output__" in bl

    def force_json_mode(self, name: str, schema_name: str = "__structured_output__") -> None:
        """Remember that a provider/model must bypass strict structured output."""
        bl_key = self._blacklist_key(name)
        self._schema_blacklist.setdefault(bl_key, set()).add(schema_name)
        self._save()

    def all_quarantined(self, names: list[str]) -> bool:
        return all(not self.is_available(n) for n in names)

    def shortest_quarantine_wait(self) -> float:
        """Seconds until the nearest quarantined model recovers."""
        now = time.time()
        waits = [s["quarantine_until"] - now for s in self._health.values() if s["quarantine_until"] > now]
        return min(waits) if waits else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Circuit Breaker — 3-State Machine (Resilience4j-inspired)
#  CLOSED → OPEN → HALF_OPEN → CLOSED
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitBreaker:
    """Three-state circuit breaker with sliding window and auto-recovery.

    - CLOSED: all requests pass; track outcomes in a sliding window.
      Trip to OPEN when failure RATE over last N >= threshold AND
      a minimum number of calls recorded (prevents 1/1 false trips).
    - OPEN: reject immediately. After wait_secs → HALF_OPEN.
    - HALF_OPEN: allow limited probes; all pass → CLOSED; any fail → OPEN.
    """

    def __init__(
        self,
        window_size: int = 50,
        min_calls: int = 10,
        failure_rate_threshold: float = 0.5,
        open_wait_secs: float = 30.0,
        half_open_probes: int = 3,
        # Legacy compat — ignored but accepted so existing code doesn't break
        max_requests: int = 80,
        max_failures: int = 15,
    ):
        self._window: deque[bool] = deque(maxlen=window_size)  # True = failure
        self._min_calls = min_calls
        self._failure_rate_threshold = failure_rate_threshold
        self._open_wait_secs = open_wait_secs
        self._max_probes = half_open_probes
        self._state = "closed"
        self._opened_at = 0.0
        self._probes_done = 0
        self._probe_failures = 0
        self._total_requests = 0
        self._total_failures = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def tripped(self) -> bool:
        """Legacy compat: True if OPEN and not yet eligible for HALF_OPEN."""
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self._open_wait_secs:
                return False  # Eligible for half-open transition
            return True
        return False

    async def allow(self) -> bool:
        """Check if a request should be allowed through."""
        async with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.monotonic() - self._opened_at >= self._open_wait_secs:
                    self._state = "half_open"
                    self._probes_done = 0
                    self._probe_failures = 0
                    logger.info("⚡ Circuit breaker → HALF_OPEN (probing)")
                    return True
                return False
            # half_open
            if self._probes_done < self._max_probes:
                self._probes_done += 1
                return True
            return False

    async def record(self, is_failure: bool) -> None:
        """Record the outcome of a request."""
        async with self._lock:
            self._total_requests += 1
            if is_failure:
                self._total_failures += 1

            if self._state == "half_open":
                if is_failure:
                    self._probe_failures += 1
                    self._trip("Half-open probe failed")
                elif self._probes_done >= self._max_probes and self._probe_failures == 0:
                    self._reset()
                return

            self._window.append(is_failure)
            if len(self._window) >= self._min_calls:
                rate = sum(self._window) / len(self._window)
                if rate >= self._failure_rate_threshold:
                    self._trip(f"Failure rate {rate:.0%} >= {self._failure_rate_threshold:.0%}")

    # Legacy sync API (backward compat for callers that haven't migrated)
    def record_success(self) -> None:
        self._total_requests += 1
        self._window.append(False)

    def record_failure(self) -> None:
        self._total_requests += 1
        self._total_failures += 1
        self._window.append(True)
        if len(self._window) >= self._min_calls:
            rate = sum(self._window) / len(self._window)
            if rate >= self._failure_rate_threshold:
                self._trip(f"Failure rate {rate:.0%} (sync)")

    def _trip(self, reason: str) -> None:
        self._state = "open"
        self._opened_at = time.monotonic()
        logger.warning("🔌 Circuit breaker → OPEN: %s", reason)

    def _reset(self) -> None:
        self._state = "closed"
        self._window.clear()
        self._probes_done = 0
        logger.info("✅ Circuit breaker → CLOSED (recovered)")

    @property
    def reason(self) -> str:
        if self._state == "open":
            remaining = max(0, self._open_wait_secs - (time.monotonic() - self._opened_at))
            return f"OPEN (recovery in {remaining:.0f}s)"
        return ""

    def status_line(self) -> str:
        fails = sum(self._window)
        total = len(self._window)
        return f"[CB:{self._state} {fails}/{total} fails, {self._total_requests} total]"


# ═══════════════════════════════════════════════════════════════════════════════
#  ModelClient wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class ModelClient:
    __slots__ = (
        "name", "client", "provider", "pipeline", "sort_priority", "credential_id",
        "critical",
    )

    def __init__(
        self,
        name: str,
        client: Any,
        provider: str,
        pipeline: str,
        sort_priority: int = 0,
        credential_id: str = "",
        critical: bool = False,
    ):
        self.name = name
        self.client = client
        self.provider = provider
        self.pipeline = pipeline
        self.sort_priority = sort_priority
        self.credential_id = credential_id
        self.critical = critical

    def __repr__(self) -> str:
        return f"ModelClient({self.name}, pipeline={self.pipeline})"


class CloudflareNativeVisionClient:
    """LangChain-compatible adapter for Workers AI's native vision route.

    Cloudflare's OpenAI-compatible endpoint serves the configured text models,
    but Llama 3.2 vision currently requires the native ``/ai/run/<model>``
    response envelope. This keeps that transport quirk inside the registry.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        model: str,
        timeout: float = 45.0,
        schema: type | None = None,
    ):
        self.account_id = account_id
        self.api_token = api_token
        self.model_name = model
        self.timeout = timeout
        self._schema = schema

    def with_structured_output(self, schema: type):
        return CloudflareNativeVisionClient(
            account_id=self.account_id,
            api_token=self.api_token,
            model=self.model_name,
            timeout=self.timeout,
            schema=schema,
        )

    @staticmethod
    def _message_payload(message: Any) -> dict[str, Any]:
        role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }.get(getattr(message, "type", ""), getattr(message, "role", "user"))
        return {"role": role, "content": getattr(message, "content", str(message))}

    async def ainvoke(self, messages: list, config: Any = None) -> Any:
        import httpx
        from langchain_core.messages import AIMessage

        payload_messages = [self._message_payload(message) for message in messages]
        if self._schema is not None:
            schema_json = json.dumps(self._schema.model_json_schema(), ensure_ascii=False)
            payload_messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "Return ONLY one valid JSON object matching this JSON Schema. "
                        "Do not use markdown fences or add commentary.\n"
                        f"SCHEMA: {schema_json}"
                    ),
                },
            )

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model_name}"
        )
        request_payload: dict[str, Any] = {
            "messages": payload_messages,
            "max_tokens": _int_env("VISION_MAX_TOKENS", 1000, 256),
            "temperature": 0.0,
        }
        if self._schema is not None:
            request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": self._schema.model_json_schema(),
            }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Cloudflare Workers AI HTTP {response.status_code}: invalid JSON response"
            ) from exc

        if response.status_code >= 400 or not body.get("success", False):
            errors = body.get("errors") or []
            detail = errors[0].get("message", "request failed") if errors else "request failed"
            raise RuntimeError(
                f"Cloudflare Workers AI HTTP {response.status_code}: "
                f"{_compact_provider_error(str(detail))}"
            )

        result = body.get("result") or {}
        content = result.get("response", "")
        if self._schema is not None:
            try:
                payload = content if isinstance(content, dict) else _extract_json_payload(content)
                return self._schema.model_validate(payload)
            except Exception:  # noqa: BLE001
                # Llama Vision occasionally ignores JSON mode when an image is
                # present even though Cloudflare lists it as supported. Repair
                # the already-grounded natural-language observation with one
                # cheap text-only turn on the same model (no image reprocessing).
                logger.debug(
                    "Cloudflare vision returned non-JSON; running text-only schema repair"
                )
                repair_payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return ONLY one valid JSON object matching this JSON Schema. "
                                "No prose or markdown.\n"
                                f"SCHEMA: {json.dumps(self._schema.model_json_schema(), ensure_ascii=False)}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Convert this visual observation into the required JSON without "
                                "adding unsupported claims:\n" + str(content)
                            ),
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": self._schema.model_json_schema(),
                    },
                    "max_tokens": _int_env("VISION_MAX_TOKENS", 1000, 256),
                    "temperature": 0.0,
                }
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    repair_response = await client.post(
                        url,
                        headers={
                            "Authorization": f"Bearer {self.api_token}",
                            "Content-Type": "application/json",
                        },
                        json=repair_payload,
                    )
                repair_body = repair_response.json()
                if repair_response.status_code >= 400 or not repair_body.get("success", False):
                    repair_errors = repair_body.get("errors") or []
                    repair_detail = (
                        repair_errors[0].get("message", "JSON repair failed")
                        if repair_errors else "JSON repair failed"
                    )
                    raise RuntimeError(
                        f"Cloudflare Workers AI HTTP {repair_response.status_code}: "
                        f"{_compact_provider_error(str(repair_detail))}"
                    )
                repaired = (repair_body.get("result") or {}).get("response", "")
                repaired_payload = (
                    repaired if isinstance(repaired, dict) else _extract_json_payload(repaired)
                )
                return self._schema.model_validate(repaired_payload)
        return AIMessage(
            content=content if isinstance(content, str) else json.dumps(content),
            response_metadata={"usage": result.get("usage", {})},
        )


# ═══════════════════════════════════════════════════════════════════════════════
def _build_text_pipeline() -> list[ModelClient]:
    """TEXT-ONLY pipeline for Manager & Helper. NEVER receives images.

    V16.1 — Multi-model-per-key architecture:
      On NVIDIA NIM, ONE API key gives access to ALL models. So we register
      multiple models under the same key. The failover loop tries every model
      on each key before moving to the next provider.
    """
    from langchain_openai import ChatOpenAI

    clients: list[ModelClient] = []
    # Structured browser actions are compact; cap completion length to keep
    # provider requests bounded.
    text_max_tokens = _int_env("TEXT_MODEL_MAX_TOKENS", 1000, 256)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    }

    # ── NVIDIA NIM — SECONDARY (multiple models, same key) ──
    # V17.0: Register the SAME top-tier model (gpt-oss-120b) here too, so the
    # model-first failover can continue with the identical model on NVIDIA when
    # all Groq keys are rate-limited. If the catalog ID is invalid on NVIDIA,
    # the startup probe prunes it automatically — zero per-step cost.
    # Override via NVIDIA_TEXT_MODELS="model1,model2" in .env.
    nvidia_keys = _collect_keys("NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEYS")
    base = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    _nvidia_models_raw = os.getenv(
        "NVIDIA_TEXT_MODELS",
        "nvidia/nemotron-3.5-lightning-30b-a3b,openai/gpt-oss-120b",
    )
    nvidia_text_models = [(m.strip(), 30) for m in _nvidia_models_raw.split(",") if m.strip()]
    # To add more NVIDIA text models, set NVIDIA_TEXT_MODELS in .env. The startup
    # capability gate auto-drops any that 404 or can't do structured output, so
    # only proven-agentic models ever reach the chain (see STRATEGY.md).
    for key_idx, key in enumerate(nvidia_keys):
        for model_name, timeout_s in nvidia_text_models:
            clients.append(
                ModelClient(
                    name=f"nvidia-text:{model_name}:{key_idx}",
                    client=ChatOpenAI(
                        model=model_name,
                        api_key=key,
                        base_url=base,
                        temperature=0.0,
                        timeout=timeout_s,
                        max_tokens=text_max_tokens,
                        default_headers=headers,
                    ),
                    provider="nvidia",
                    pipeline="text",
                    credential_id=_credential_fingerprint(key),
                )
            )
    if nvidia_keys:
        logger.info(
            "TEXT ── NVIDIA NIM (SECONDARY): %d keys × %d models = %d instances",
            len(nvidia_keys),
            len(nvidia_text_models),
            len(nvidia_keys) * len(nvidia_text_models),
        )

    # ── Cloudflare Workers AI — OpenAI-compatible free allocation ──
    # Requires both an account id and an API token. Multiple models share the
    # same account-level neuron allocation, so model diversity is useful for
    # capability/latency failover but does not multiply quota.
    cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    cloudflare_tokens = _collect_keys(
        "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_TOKENS"
    )
    if _env_flag("CLOUDFLARE_ENABLED", True) and cloudflare_tokens:
        if not cloudflare_account_id:
            logger.warning(
                "TEXT ── Cloudflare token configured without CLOUDFLARE_ACCOUNT_ID; skipping"
            )
        else:
            cloudflare_base = os.getenv("CLOUDFLARE_BASE_URL", "").strip() or (
                f"https://api.cloudflare.com/client/v4/accounts/"
                f"{cloudflare_account_id}/ai/v1"
            )
            cloudflare_models = [
                m.strip()
                for m in os.getenv(
                    "CLOUDFLARE_TEXT_MODELS",
                    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                ).split(",")
                if m.strip()
            ]
            for key_idx, token in enumerate(cloudflare_tokens):
                for model_name in cloudflare_models:
                    clients.append(
                        ModelClient(
                            name=f"cloudflare:{model_name}:{key_idx}",
                            client=ChatOpenAI(
                                model=model_name,
                                api_key=token,
                                base_url=cloudflare_base,
                                temperature=0.0,
                                timeout=30,
                                max_tokens=min(
                                    _int_env("CLOUDFLARE_MAX_TOKENS", 2048, 256),
                                    text_max_tokens,
                                ),
                                default_headers=headers,
                            ),
                            provider="cloudflare",
                            pipeline="text",
                            credential_id=_credential_fingerprint(cloudflare_account_id),
                        )
                    )
            logger.info(
                "TEXT ── Cloudflare Workers AI: %d token(s) × %d models = %d instances",
                len(cloudflare_tokens),
                len(cloudflare_models),
                len(cloudflare_tokens) * len(cloudflare_models),
            )

    # ── Google Gemini — primary worker (role ordering is applied below) ──
    gemini_text_keys = _collect_keys("GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACKS")
    if gemini_text_keys:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            gemini_text_model = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite")
            for idx, key in enumerate(gemini_text_keys):
                clients.append(
                    ModelClient(
                        name=f"gemini-text:{gemini_text_model}:{idx}",
                        client=ChatGoogleGenerativeAI(
                            model=gemini_text_model,
                            google_api_key=key,
                            temperature=0.0,
                            timeout=30,
                            max_output_tokens=text_max_tokens,
                        ),
                        provider="google",
                        pipeline="text",
                        credential_id=_credential_fingerprint(key),
                    )
                )
            logger.info(
                "TEXT ── Google Gemini: %d keys loaded (%s)",
                len(gemini_text_keys),
                gemini_text_model,
            )
        except ImportError:
            logger.warning("TEXT ── langchain-google-genai not installed, skipping Gemini text fallback")
        except Exception as e:
            logger.error("TEXT ── Gemini text fallback bootstrap failed: %s", e)

    return clients


# ═══════════════════════════════════════════════════════════════════════════════
#  VISION Pipeline — Google Gemini (proven) + NVIDIA NIM Vision (fallback)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_vision_pipeline() -> list[ModelClient]:
    """VISION-ONLY pipeline for Executor. ALWAYS receives images."""
    clients: list[ModelClient] = []

    # ── Google Gemini — PRIMARY VISION (2 keys, gemma-4 vision model) ──
    gemini_keys = _collect_keys(
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_FALLBACKS",
        "VISION_GOOGLE_API_KEY",
        "VISION_GOOGLE_API_KEY_FALLBACKS",
        "GOOGLE_API_KEY",
    )
    if gemini_keys:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("WORKER_VLM_MODEL", "gemma-4-31b-it")
            for idx, key in enumerate(gemini_keys):
                clients.append(
                    ModelClient(
                        name=f"gemini-vision:{model}:{idx}",
                        client=ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=key,
                            temperature=0.0,
                            timeout=45,
                        ),
                        provider="google",
                        pipeline="vision",
                        credential_id=_credential_fingerprint(key),
                    )
                )
            logger.info("VISION ── Google Gemini (PRIMARY): %d keys loaded (%s)", len(gemini_keys), model)
        except ImportError:
            logger.warning("VISION ── langchain-google-genai not installed, skipping Gemini")
        except Exception as e:
            logger.error("VISION ── Gemini bootstrap failed: %s", e)

    # ── NVIDIA NIM Vision — SECONDARY (multi-model × ALL NVIDIA keys) ──
    # V16.1: Collect ALL NVIDIA keys so every vision model gets tried on every key
    nvidia_vision_keys = _collect_keys(
        "NVIDIA_VISION_API_KEY",
        "NVIDIA_NIM_API_KEY",
        "NVIDIA_NIM_API_KEYS",
    )
    if nvidia_vision_keys:
        from langchain_openai import ChatOpenAI

        base = os.getenv("NVIDIA_VISION_BASE_URL", "https://integrate.api.nvidia.com/v1")

        # OPTIONAL + env-configurable via NVIDIA_VISION_MODELS (comma-separated),
        # mirroring NVIDIA_TEXT_MODELS. Default leads with Llama 4 Maverick (the
        # chosen vision model); the rest are optional fallbacks. All probed
        # 2026-06 to respond <1s; the startup capability gate drops any that 404
        # or can't structure. To use ONLY Maverick, set
        # NVIDIA_VISION_MODELS="meta/llama-4-maverick-17b-128e-instruct".
        _nv_vision_raw = os.getenv(
            "NVIDIA_VISION_MODELS",
            "meta/llama-4-maverick-17b-128e-instruct,"  # Llama 4 Maverick — primary
            "meta/llama-3.2-11b-vision-instruct,"  # fast, lightweight fallback
            "nvidia/llama-3.1-nemotron-nano-vl-8b-v1,"  # fast nano VL fallback
            "meta/llama-3.2-90b-vision-instruct",  # strong 90B fallback
        )
        _nv_vision_timeout = int(os.getenv("NVIDIA_VISION_TIMEOUT", "30"))
        nvidia_vision_models = [
            (m.strip(), _nv_vision_timeout) for m in _nv_vision_raw.split(",") if m.strip()
        ]
        for key_idx, key in enumerate(nvidia_vision_keys):
            for model_name, timeout_s in nvidia_vision_models:
                clients.append(
                    ModelClient(
                        name=f"nvidia-vision:{model_name}:{key_idx}",
                        client=ChatOpenAI(
                            model=model_name,
                            api_key=key,
                            base_url=base,
                            temperature=0.0,
                            timeout=timeout_s,
                        ),
                        provider="nvidia",
                        pipeline="vision",
                        credential_id=_credential_fingerprint(key),
                    )
                )
        logger.info(
            "VISION ── NVIDIA NIM (SECONDARY): %d keys × %d models = %d instances",
            len(nvidia_vision_keys),
            len(nvidia_vision_models),
            len(nvidia_vision_keys) * len(nvidia_vision_models),
        )

    # ── Cloudflare Workers AI vision — optional, account-level free allocation ──
    cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    cloudflare_tokens = _collect_keys(
        "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_TOKENS"
    )
    if (_env_flag("CLOUDFLARE_ENABLED", True)
            and _env_flag("CLOUDFLARE_VISION_ENABLED", True)
            and cloudflare_account_id and cloudflare_tokens):
        cloudflare_vision_models = [
            m.strip()
            for m in os.getenv(
                "CLOUDFLARE_VISION_MODELS",
                "@cf/meta/llama-3.2-11b-vision-instruct",
            ).split(",")
            if m.strip()
        ]
        for key_idx, token in enumerate(cloudflare_tokens):
            for model_name in cloudflare_vision_models:
                clients.append(
                    ModelClient(
                        name=f"cloudflare-vision:{model_name}:{key_idx}",
                        client=CloudflareNativeVisionClient(
                            account_id=cloudflare_account_id,
                            api_token=token,
                            model=model_name,
                            timeout=45,
                        ),
                        provider="cloudflare",
                        pipeline="vision",
                        credential_id=_credential_fingerprint(cloudflare_account_id),
                    )
                )
        logger.info(
            "VISION ── Cloudflare Workers AI: %d token(s) × %d models = %d instances",
            len(cloudflare_tokens),
            len(cloudflare_vision_models),
            len(cloudflare_tokens) * len(cloudflare_vision_models),
        )

    return clients


# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO Pipeline — bounded calls for detected survey media questions
# ═══════════════════════════════════════════════════════════════════════════════


def _build_audio_pipeline() -> list[ModelClient]:
    """AUDIO-ONLY Gemini chain, invoked solely for detected media questions."""
    if not _env_flag("SURVEY_AUDIO_ENABLED", True):
        return []
    keys = _collect_keys(
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_FALLBACKS",
        "GOOGLE_API_KEY",
        "GOOGLE_API_KEY_FALLBACKS",
        "VISION_GOOGLE_API_KEY",
        "VISION_GOOGLE_API_KEY_FALLBACKS",
    )
    if not keys:
        return []
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv(
            "SURVEY_AUDIO_MODEL",
            os.getenv("WORKER_VLM_MODEL", "gemini-3.5-flash"),
        ).strip()
        clients = [
            ModelClient(
                name=f"gemini-audio:{model}:{idx}",
                client=ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=key,
                    temperature=0.0,
                    timeout=35,
                ),
                provider="google",
                pipeline="audio",
                credential_id=_credential_fingerprint(key),
            )
            for idx, key in enumerate(keys)
        ]
        logger.info("AUDIO ── Google Gemini: %d keys loaded (%s)", len(keys), model)
        return clients
    except Exception as exc:  # noqa: BLE001 - absence falls back to a constrained guess
        logger.warning("AUDIO ── Gemini bootstrap unavailable: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
#  PREMIUM Pipeline — one paid key/model serves BOTH text and vision (V24)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_premium_pipeline() -> tuple[list["ModelClient"], list["ModelClient"]]:
    """Premium mode: a single top-tier multimodal model drives everything.

    Trusted by definition — no probe, no capability gate, no free-tier fallback
    juggling. OpenAI-compatible by default (covers OpenAI / OpenRouter / Together
    / Fireworks / any gateway); PREMIUM_PROVIDER=google uses the Gemini client.
    Returns (text_clients, vision_clients); audio remains a separate pipeline.
    """
    cfg = _premium_config()
    keys, model = cfg["keys"], cfg["model"]
    if not keys or not model:
        logger.error("PREMIUM mode needs PREMIUM_API_KEY and PREMIUM_MODEL — none/partial set.")
        return [], []

    def _mk(m: str, key: str, idx: int, pipeline: str) -> "ModelClient":
        if cfg["provider"] == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            client = ChatGoogleGenerativeAI(
                model=m, google_api_key=key, temperature=0.0, timeout=cfg["timeout"]
            )
        else:
            from langchain_openai import ChatOpenAI

            client = ChatOpenAI(
                model=m, api_key=key, base_url=cfg["base_url"], temperature=0.0, timeout=cfg["timeout"]
            )
        return ModelClient(
            name=f"premium-{pipeline}:{m}:{idx}", client=client,
            provider="premium", pipeline=pipeline,
            credential_id=_credential_fingerprint(key),
        )

    text = [_mk(model, k, i, "text") for i, k in enumerate(keys)]
    vmodel = cfg["vision_model"] or model
    vision = [_mk(vmodel, k, i, "vision") for i, k in enumerate(keys)]
    logger.info(
        "⭐ PREMIUM mode: text='%s', vision='%s' on %d key(s) — no probe, no gate", model, vmodel, len(keys)
    )
    return text, vision


def _build_premium_audio_pipeline() -> list["ModelClient"]:
    """Use the premium Google key for audio without changing the legacy tuple API."""
    cfg = _premium_config()
    if (cfg["provider"] != "google" or not cfg["keys"]
            or not _env_flag("SURVEY_AUDIO_ENABLED", True)):
        return []
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv(
        "SURVEY_AUDIO_MODEL",
        cfg["vision_model"] or cfg["model"],
    ).strip()
    return [
        ModelClient(
            name=f"premium-audio:{model}:{idx}",
            client=ChatGoogleGenerativeAI(
                model=model,
                google_api_key=key,
                temperature=0.0,
                timeout=cfg["timeout"],
            ),
            provider="premium",
            pipeline="audio",
            credential_id=_credential_fingerprint(key),
        )
        for idx, key in enumerate(cfg["keys"])
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  ModelRegistry Singleton
# ═══════════════════════════════════════════════════════════════════════════════


class ModelRegistry:
    """Centralized registry with strictly separated text/vision/audio clients."""

    _instance: "ModelRegistry | None" = None

    def __init__(self):
        # V24 dual-mode: premium (one trusted paid key) vs free (juggle + gate).
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
        """V24 role separation — the WORKER (action decisions, the critical path)
        draws only from the most capable models. Premium → the single premium
        model. Free → tier ≤ WORKER_MAX_TIER, ordered by WORKER_MODEL_ORDER.
        Chronically unreliable instances are omitted so the worker fails fast
        instead of paying their timeout tax on every survey question."""
        if self.mode == "premium":
            return list(self._text_pipeline)
        top = [mc for mc in self._text_pipeline if get_model_tier(mc.name) <= WORKER_MAX_TIER]
        selected = top or list(self._text_pipeline)
        reliable = [
            mc for mc in selected
            if not (
                (mc.credential_id or getattr(self.health, "_persistence_path", None) is None)
                and self.health.is_chronically_unreliable(mc.name)
            )
        ]
        if reliable:
            selected = reliable
        elif selected:
            logger.error(
                "All configured worker models are chronically unreliable; "
                "worker calls will fail fast until a startup probe succeeds."
            )
            selected = []
        prioritized = [
            ModelClient(
                name=mc.name,
                client=mc.client,
                provider=mc.provider,
                pipeline=mc.pipeline,
                sort_priority=_worker_priority(mc),
                credential_id=mc.credential_id,
                critical=True,
            )
            for mc in selected
        ]
        return order_failover_chain(prioritized, self.health)

    def get_worker_chain_names(self) -> list[str]:
        return [mc.name for mc in self.get_worker_chain()]

    def get_auxiliary_chain(self) -> list[ModelClient]:
        """Return a provider-prioritized chain for high-volume support calls.

        The worker still uses capability tiers. Planner, PRM, WebDreamer,
        reality reconciliation and the outcome judge instead prefer providers
        intended for high-volume free inference, keeping the critical browser
        decision path isolated.
        """
        if self.mode == "premium":
            return list(self._text_pipeline)
        requested = [
            p.strip().lower()
            for p in os.getenv(
                "AUXILIARY_PROVIDER_ORDER",
                "google,cloudflare,nvidia",
            ).split(",")
            if p.strip()
        ]
        rank = {provider: idx for idx, provider in enumerate(requested)}
        fallback_rank = len(rank)
        return [
            ModelClient(
                name=mc.name,
                client=mc.client,
                provider=mc.provider,
                pipeline=mc.pipeline,
                sort_priority=rank.get(mc.provider.lower(), fallback_rank),
                credential_id=mc.credential_id,
            )
            for mc in self._text_pipeline
        ]

    def get_auxiliary_chain_names(self) -> list[str]:
        chain = self.get_auxiliary_chain()
        return [mc.name for mc in order_failover_chain(chain, self.health)]

    def _apply_capability_gate(
        self,
        pipeline: list[ModelClient],
        dead: set,
        incapable: set,
        label: str,
    ) -> list[ModelClient]:
        """Remove dead + incapable combos. SAFETY: if that empties the pipeline,
        restore the alive (non-dead) combos — preferring allowlisted ones — so a
        pipeline is NEVER empty (invariant #2)."""
        kept = [mc for mc in pipeline if _combo_of(mc) not in dead and _combo_of(mc) not in incapable]
        if kept:
            if len(kept) != len(pipeline):
                logger.info("🔬 Capability gate [%s]: %d → %d models", label, len(pipeline), len(kept))
            return kept
        alive = [mc for mc in pipeline if _combo_of(mc) not in dead]
        allow = [
            mc
            for mc in alive
            if normalize_model_id(ProviderHealthTracker._base_model_name(mc.name)) in AGENTIC_TEXT_ALLOWLIST
        ]
        floor = allow or alive or pipeline
        logger.warning("🛟 Capability gate would EMPTY %s — restoring %d floor model(s)", label, len(floor))
        return floor

    def summary(self) -> str:
        t = " → ".join(self.get_text_chain_names()) or "(none)"
        v = " → ".join(self.get_vision_chain_names()) or "(none)"
        return (
            f"TEXT chain ({len(self._text_pipeline)}): {t}\nVISION chain ({len(self._vision_pipeline)}): {v}"
        )

    # ── V17.0: Startup probe — prune dead models, seed latency ──────────────

    async def probe_and_prune(
        self, timeout: float = 8.0, vision_timeout: float | None = None,
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

        from langchain_core.messages import HumanMessage

        # One representative instance per (provider, base_model, pipeline)
        reps: dict[tuple[str, str, str], ModelClient] = {}
        for mc in self._text_pipeline + self._vision_pipeline:
            key = (
                mc.provider,
                normalize_model_id(ProviderHealthTracker._base_model_name(mc.name)),
                mc.pipeline,
            )
            current = reps.get(key)
            # Do not repeatedly probe a chronically bad key when a healthy
            # sibling represents the same provider/model combination. The old
            # first-key selection made Gemini key 0's history look like a
            # Gemini vision initialization failure on every fresh run.
            if current is None or (
                self.health.is_chronically_unreliable(current.name)
                and not self.health.is_chronically_unreliable(mc.name)
            ):
                reps[key] = mc

        cached_count = sum(
            1 for mc in reps.values() if self.health.probe_cache_fresh(mc.name)
        )
        reps = {
            key: mc for key, mc in reps.items()
            if not self.health.probe_cache_fresh(mc.name)
        }
        if cached_count:
            logger.info("🔬 Capability probe cache reused for %d healthy combo(s)", cached_count)

        if not reps:
            return

        # V24 CAPABILITY probe (not just liveness): each model must produce a
        # valid structured object for a tiny agentic task. Alive-but-can't-
        # structure models (e.g. gemma-4-31b-it on NVIDIA) are EXCLUDED — that's
        # the agentic capability gate. See STRATEGY.md §2.
        from pydantic import BaseModel as _BM, Field as _F
        from langchain_core.messages import SystemMessage

        class _CapProbe(_BM):
            action: str = _F(description="one of: click, type, scroll, done")
            element_id: str = _F(description="an element id like e5")
            confidence: float = _F(description="0..1")

        cap_messages = [
            SystemMessage(
                content="You are a browser agent. Choose the next action and "
                "reply with the structured fields."
            ),
            HumanMessage(
                content="Page elements: [e3] search box (empty), [e7] Submit "
                "button. Goal: type 'running shoes' into the search box. "
                "Give your next action."
            ),
        ]

        _DEAD = ("404", "401", "410")
        _DEAD_L = (
            "not_found",
            "does not exist",
            "model_not_found",
            "invalid api key",
            "invalid_api_key",
            "unauthorized",
            "decommissioned",
            "gone",
        )

        def _kind(exc) -> str:
            s = str(exc)
            sl = s.lower()
            if any(d in s for d in _DEAD) or any(d in sl for d in _DEAD_L):
                return "dead"
            if "429" in s or "rate limit" in sl or "rate_limit" in sl:
                return "transient"
            return "other"

        async def _probe(combo_key, mc: ModelClient):
            probe_timeout = (
                vision_timeout if mc.pipeline == "vision" and vision_timeout is not None
                else timeout
            )
            t0 = time.monotonic()
            try:  # 1) strict structured output — success ⇒ the model CAN structure
                await asyncio.wait_for(
                    mc.client.with_structured_output(_CapProbe).ainvoke(cap_messages), timeout=probe_timeout
                )
                return combo_key, time.monotonic() - t0, "capable", "strict"
            except asyncio.TimeoutError:
                return combo_key, time.monotonic() - t0, "timeout", None
            except Exception as exc:
                k = _kind(exc)
                if k == "dead":
                    return combo_key, time.monotonic() - t0, "dead", str(exc)[:120]
                if k == "transient":
                    return combo_key, time.monotonic() - t0, "transient", str(exc)[:100]
                try:  # 2) strict rejected the schema — does JSON-mode rescue work?
                    await asyncio.wait_for(
                        _invoke_json_mode(mc.client, cap_messages, _CapProbe, probe_timeout), timeout=probe_timeout
                    )
                    return combo_key, time.monotonic() - t0, "capable", "json-mode"
                except asyncio.TimeoutError:
                    return combo_key, time.monotonic() - t0, "timeout", None
                except Exception as e2:
                    if _kind(e2) == "transient":
                        return combo_key, time.monotonic() - t0, "transient", str(e2)[:100]
                    # Responded but cannot structure via EITHER path → INCAPABLE.
                    return combo_key, time.monotonic() - t0, "incapable", str(e2)[:100]

        logger.info("🔬 Capability-probing %d (provider, model) combos (≤%.0fs)...", len(reps), timeout)
        results = await asyncio.gather(*[_probe(k, mc) for k, mc in reps.items()])

        dead_combos: set = set()
        incapable_combos: set = set()
        timed_out_instances: set[str] = set()
        for combo_key, elapsed, status, detail in results:
            provider, base, pipeline = combo_key
            if status == "capable":
                logger.info(
                    "✅ Capable: %s/%s (%s) — %.1fs%s",
                    provider,
                    base,
                    pipeline,
                    elapsed,
                    " [json-mode]" if detail == "json-mode" else "",
                )
                latency_targets = (
                    [reps[combo_key]]
                    if provider == "google"
                    else [
                        mc for mc in self._text_pipeline + self._vision_pipeline
                        if _combo_of(mc) == combo_key
                    ]
                )
                for mc in latency_targets:
                    if _combo_of(mc) == combo_key:
                        self.health.record_success(mc.name, latency=elapsed)
                if detail == "json-mode":
                    # Schema support belongs to the provider/model endpoint,
                    # not one credential, so this remains family-wide.
                    self.health.force_json_mode(reps[combo_key].name)
            elif status == "dead":
                dead_combos.add(combo_key)
                logger.warning("💀 Prune DEAD: %s/%s (%s) — %s", provider, base, pipeline, detail)
            elif status == "incapable":
                incapable_combos.add(combo_key)
                logger.warning(
                    "🚫 Prune INCAPABLE (can't structure): %s/%s (%s) — %s", provider, base, pipeline, detail
                )
            elif status == "timeout":
                probe_timeout = (
                    vision_timeout if pipeline == "vision" and vision_timeout is not None
                    else timeout
                )
                logger.warning(
                    "🐢 Startup probe timeout: %s/%s (%s) — timed out at %.0fs; retaining in chain",
                    provider, base, pipeline, probe_timeout,
                )
                timeout_targets = (
                    [reps[combo_key]]
                    if provider == "google"
                    else [
                        mc for mc in self._text_pipeline + self._vision_pipeline
                        if _combo_of(mc) == combo_key
                    ]
                )
                for mc in timeout_targets:
                    # A startup probe is a best-effort capability check. A
                    # cold provider response must not count as a real failed
                    # browser call or poison the persistent health record.
                    self.health.record_failure(
                        mc.name, latency=probe_timeout, reliability_failure=False,
                    )
                    timed_out_instances.add(mc.name)
            else:  # transient
                logger.info("ℹ️ Transient (kept): %s/%s (%s) — %s", provider, base, pipeline, detail)

        if dead_combos or incapable_combos:
            bt, bv = len(self._text_pipeline), len(self._vision_pipeline)
            self._text_pipeline = self._apply_capability_gate(
                self._text_pipeline, dead_combos, incapable_combos, "TEXT"
            )
            self._vision_pipeline = self._apply_capability_gate(
                self._vision_pipeline, dead_combos, incapable_combos, "VISION"
            )
            logger.info(
                "🔬 Capability gate — %d dead, %d incapable | TEXT %d→%d, VISION %d→%d",
                len(dead_combos),
                len(incapable_combos),
                bt,
                len(self._text_pipeline),
                bv,
                len(self._vision_pipeline),
            )
        elif not timed_out_instances:
            logger.info("🔬 Capability probe complete — all %d combos capable", len(reps))

        if timed_out_instances:
            # A startup timeout is not proof that a model/key is unavailable.
            # Keep it in the pipeline so the normal per-key health-aware
            # rotation can retry it after cooldown and so sibling vision models
            # remain a real fallback chain. Previously this deleted the
            # representative instance for the whole session, making a transient
            # Gemini probe delay look like vision initialization failure.
            logger.warning(
                "🐢 Startup probe timed out for %d instance(s); retaining them without persistent failure penalties",
                len(timed_out_instances),
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Failover Invocation — V17.0 model-first ordering
# ═══════════════════════════════════════════════════════════════════════════════


def _as_model_clients(chain: list) -> list[ModelClient]:
    """Normalize chain entries to ModelClient (accepts raw LLM clients too)."""
    entries: list[ModelClient] = []
    for item in chain:
        if isinstance(item, ModelClient):
            entries.append(item)
        else:
            name = getattr(item, "model_name", getattr(item, "model", str(item)))
            entries.append(ModelClient(name=str(name), client=item, provider="unknown", pipeline="text"))
    return entries


def order_failover_chain(
    entries: list[ModelClient],
    health: ProviderHealthTracker | None,
) -> list[ModelClient]:
    """Sort by a role-biased expected time-to-answer score.

    All instances of the best model group at the front — every key, every
    provider hosting it — so failover rotates keys/providers for the SAME
    model before ever falling to a weaker one. Role-specific chains can assign
    a provider priority (for example, Cloudflare first for auxiliary traffic).
    Role order remains the cold-start preference, but it is deliberately a
    finite penalty rather than a hard wall. A proven fast fallback can lead the
    next run after the nominal primary repeatedly times out.
    """

    def sort_key(mc: ModelClient):
        tier = get_model_tier(mc.name)
        # Hand-built/test clients have no credential identity.  Never attach a
        # persisted real key's stale health to such a client merely because its
        # display name happens to match; ephemeral trackers remain fully usable.
        use_health = bool(
            health
            and (mc.credential_id or getattr(health, "_persistence_path", None) is None)
        )
        cost = health.expected_cost(mc.name) if use_health else 0.0
        quota = health.quota_penalty(mc.name) if use_health else 0.0
        role_penalty = _float_env(
            "MODEL_ROLE_PRIORITY_PENALTY_SECONDS", 6.0, minimum=0.0
        )
        tier_penalty = _float_env(
            "MODEL_TIER_PENALTY_SECONDS", 5.0, minimum=0.0
        )
        score = cost + quota + mc.sort_priority * role_penalty + tier * tier_penalty
        return (score, tier, mc.sort_priority)

    return sorted(entries, key=sort_key)


def _extract_text(raw_response: Any) -> str:
    """Normalize provider response objects into plain text."""
    content = getattr(raw_response, "content", raw_response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _estimate_input_tokens(messages: list, schema: type | None = None) -> int:
    """Cheap local token estimate used only for per-project quota scheduling."""
    characters = 0
    for message in messages or []:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    # Never count/log base64 image payload bytes as text tokens.
                    if isinstance(part.get("text"), str):
                        characters += len(part["text"])
                elif isinstance(part, str):
                    characters += len(part)
        else:
            characters += len(str(content))
    if schema is not None:
        try:
            characters += len(json.dumps(schema.model_json_schema()))
        except Exception:
            pass
    return max(1, (characters + 3) // 4)


def _extract_json_payload(text: str) -> dict:
    """Extract the first JSON object from model output (handles ``` fences)."""
    candidate = text.strip()
    if "```json" in candidate:
        candidate = candidate.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in candidate:
        candidate = candidate.split("```", 1)[1].split("```", 1)[0]
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(candidate[start : end + 1])


def _compact_provider_error(error: str, limit: int = 280) -> str:
    """Keep quota diagnostics useful while ensuring credentials never reach logs."""
    compact = re.sub(r"\s+", " ", error).strip()
    compact = re.sub(
        r"\b(?:gsk_|nvapi-|csk-|cfut_|sk-|AIza)[A-Za-z0-9_.-]+",
        "<redacted-key>",
        compact,
    )
    return compact[:limit] or "provider returned HTTP 429"


def _retry_after_seconds(error: str) -> float | None:
    """Extract common Retry-After/try-again durations embedded in SDK errors."""
    match = re.search(
        r"(?:retry[-_ ]?after|retry[-_ ]?delay|try again in)\D{0,20}"
        r"([0-9]+(?:\.[0-9]+)?)\s*"
        r"(ms|milliseconds?|s|sec(?:ond)?s?|m|min(?:ute)?s?|h|hours?)?",
        error,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    if unit.startswith("ms"):
        value /= 1000.0
    elif unit.startswith("m"):
        value *= 60.0
    elif unit.startswith("h"):
        value *= 3600.0
    return min(value, 86400.0)


def _drain_cancelled_task(task: asyncio.Task) -> None:
    """Consume a detached provider task's eventual exception/cancellation."""
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


async def _hard_timeout(awaitable: Any, timeout: float) -> Any:
    """Return at the deadline without awaiting a provider's slow cancellation.

    ``asyncio.wait_for`` waits for the underlying coroutine to acknowledge
    cancellation. Some SDKs wrap synchronous HTTP clients and can take minutes
    (or longer) to do that. A detached task is cancelled and drained in the
    background, while the failover caller immediately receives TimeoutError and
    can enforce its total budget.
    """
    task = asyncio.ensure_future(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=max(0.001, float(timeout)))
    if task not in done:
        task.cancel()
        task.add_done_callback(_drain_cancelled_task)
        raise asyncio.TimeoutError
    if task.cancelled():
        raise asyncio.TimeoutError
    return task.result()


async def _invoke_json_mode(llm, messages: list, schema: type, timeout: float) -> Any:
    """Plain JSON-mode fallback for providers that reject the strict schema.

    Appends the JSON Schema as an instruction, parses the raw reply, and
    validates it into the Pydantic model — so a top-tier model that fails
    structured output still serves instead of being benched.
    """
    from langchain_core.messages import HumanMessage

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    instruction = HumanMessage(
        content=(
            "Respond ONLY with a single JSON object that matches this JSON Schema. "
            "No markdown fences, no commentary, no extra keys.\n"
            f"SCHEMA: {schema_json}"
        )
    )
    raw = await _hard_timeout(llm.ainvoke(list(messages) + [instruction]), timeout)
    payload = _extract_json_payload(_extract_text(raw))
    return schema.model_validate(payload)


def _compact_messages_for_retry(messages: list, max_chars: int = 9000) -> list:
    """Make a bounded retry request while preserving the current objective/state.

    Providers commonly return HTTP 413 when the prompt plus reserved output
    exceeds a free-tier request limit.  Retrying the same bytes only burns
    another key; retain the system instructions and the tail of the latest
    user message, where the live page/action context is anchored.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if not messages:
        return messages
    remaining = max(2000, int(max_chars))
    compacted = []
    for index, message in enumerate(messages):
        content = getattr(message, "content", message)
        if not isinstance(content, str):
            compacted.append(message)
            continue
        reserve = 2400 if isinstance(message, SystemMessage) else 6000
        if index == len(messages) - 1:
            reserve = min(remaining, max(2000, reserve))
            text = content[-reserve:]
            if len(content) > reserve:
                text = "[earlier context compacted]\n" + text
        else:
            reserve = min(remaining, reserve)
            text = content[:reserve]
            if len(content) > reserve:
                text += "\n[remaining context compacted]"
        cls = SystemMessage if isinstance(message, SystemMessage) else HumanMessage
        compacted.append(cls(content=text))
        remaining -= len(text)
        if remaining <= 0:
            break
    return compacted


async def invoke_with_failover(
    chain: list,
    messages: list,
    schema: type | None = None,
    *,
    breaker: CircuitBreaker | None = None,
    health: ProviderHealthTracker | None = None,
    timeout_seconds: float = 20.0,
    total_timeout_seconds: float | None = None,
    timeout_cooldown_seconds: float = 30.0,
    timeout_sibling_threshold: int = 2,
    health_tracker: "ProviderHealthTracker | None" = None,
    base64_image: str | None = None,
    role: str | None = None,
    max_attempts: int | None = None,
) -> tuple[Any, str]:
    """V17.0 — Model-first failover with adaptive timeouts.

    Ordering: (model quality tier, E[cost]) — ALL instances of the best model
    (across every API key and provider) are exhausted before any weaker model.

    Per-instance behavior:
      - 429 → jittered cooldown on THAT instance only; the very next attempt
        is the SAME model on the next independent key/provider (no
        provider-wide skip and no sleep).
      - Timeout → adaptive, with a production floor configured by
        ``MODEL_TIMEOUT_FLOOR_SECONDS``. The caller's ``timeout_seconds`` remains
        a hard cap, so deliberately cheap consensus voters can still opt into a
        smaller budget. Repeated hangs cool only the affected key/project.
      - ``total_timeout_seconds`` places a hard wall-clock budget around the
        entire sequential failover walk.
      - Schema 400 → blacklist the (provider, model, schema) combo AND rescue
        the call immediately via JSON-mode on the same model.
      - Blacklisted combos are not skipped — they run in JSON mode, so a
        top-tier model never gets benched over a schema quirk.
    """
    # Accept either kwarg name for backward compatibility
    _health = health or health_tracker

    # A missing total budget used to mean an unbounded sequential walk. Keep
    # explicit caller budgets intact, but cap all ordinary worker/auxiliary
    # calls at a bounded wall-clock budget from the environment.
    if total_timeout_seconds is None:
        total_timeout_seconds = MODEL_FAILOVER_BUDGET_SECONDS
    requested_max_attempts = max_attempts

    if breaker and breaker.tripped:
        raise RuntimeError(f"Circuit breaker tripped: {breaker.reason}")

    entries = _as_model_clients(chain)
    if _health:
        benched = [
            mc for mc in entries
            if (
                (mc.credential_id or getattr(_health, "_persistence_path", None) is None)
                and _health.is_chronically_unreliable(mc.name)
            )
        ]
        entries = [
            mc for mc in entries
            if not (
                (mc.credential_id or getattr(_health, "_persistence_path", None) is None)
                and _health.is_chronically_unreliable(mc.name)
            )
        ]
        if benched:
            logger.warning(
                "🪑 FAILOVER: benched %d chronically unreliable model instance(s)",
                len(benched),
            )
    if not entries:
        raise RuntimeError("No reliable model instances are currently available")
    ordered = order_failover_chain(entries, _health)
    attempt_role = role or ("VISION" if base64_image else "TEXT_WORKER")
    max_attempts = max(1, int(
        requested_max_attempts
        or (ROLE_MAX_ATTEMPTS.get(attempt_role.upper(), MODEL_FAILOVER_MAX_ATTEMPTS)
            if role is not None else MODEL_FAILOVER_MAX_ATTEMPTS)
    ))
    if _health:
        preferred = _health.preferred_for_role(attempt_role, {mc.name for mc in ordered})
        if preferred:
            ordered = [next(mc for mc in ordered if mc.name == preferred)] + [
                mc for mc in ordered if mc.name != preferred
            ]
    deadline = (
        time.monotonic() + total_timeout_seconds
        if total_timeout_seconds is not None and total_timeout_seconds > 0
        else None
    )

    # Cooled instances stay parked. Retrying known timeouts/429s on every worker
    # decision consumed most of the previous run after healthy quota ran out.
    if _health:
        quota_guarded = {
            mc.name
            for mc in ordered
            if _health.in_quota_guard(mc.name, critical=mc.critical)
        }
        hot = [
            mc for mc in ordered
            if mc.name not in quota_guarded
            and not _health.in_cooldown(mc.name, critical=mc.critical)
        ]
        cooled = [
            mc for mc in ordered
            if mc.name not in quota_guarded
            and _health.in_cooldown(mc.name, critical=mc.critical)
        ]
        if quota_guarded:
            logger.info(
                "🧮 QUOTA GUARD: parked %d key/model instance(s) at their configured free-limit threshold",
                len(quota_guarded),
            )
    else:
        hot, cooled = ordered, []
    allow_cooled = _env_flag("MODEL_ALLOW_COOLED_LAST_RESORT", False)
    passes = [hot] if hot else []
    if allow_cooled and cooled:
        passes.append(cooled)
    if not passes:
        raise RuntimeError("All reliable model instances are cooling down or quota-guarded")

    schema_name = getattr(schema, "__name__", "__structured_output__") if schema is not None else ""
    last_error: str = ""
    attempt_no = 0
    attempted_providers: set[str] = set()
    cross_provider_rescues = 0
    total = len(entries)
    budget_exhausted = False
    oversized_scopes: set[str] = set()
    timed_out_families: set[str] = set()
    timeout_strikes: dict[str, int] = {}
    provider_timeout_strikes: dict[str, int] = {}
    estimated_input_tokens = _estimate_input_tokens(messages, schema)
    def attempt_event(mc: ModelClient, attempt: int, started: float, *, classification: str,
                      metadata: dict[str, Any] | None = None, repair_attempted: bool = False,
                      repair_type: str = "", ultimately_used: bool = False) -> None:
        details = metadata or {}
        event = {
            "run_id": PROCESS_RUN_ID, "timestamp": datetime.now().isoformat(),
            "provider": mc.provider, "model": mc.name, "role": attempt_role,
            "credential": mc.credential_id or "anonymous", "attempt": attempt,
            "timeout": round(timeout, 3), "elapsed_ms": round((time.monotonic() - started) * 1000),
            "http_status": details.get("http_status"), "exception_class": details.get("exception_class", ""),
            "classification": classification, "response_size": details.get("response_size", 0),
            "response_hash": details.get("response_hash", ""),
            "sanitized_diagnostic": details.get("diagnostic", "")[:500],
            "repair_attempted": repair_attempted, "repair_type": repair_type,
            "ultimately_used": ultimately_used,
        }
        logger.info("MODEL_ATTEMPT_EVENT %s", json.dumps(event, separators=(",", ":"), ensure_ascii=False))

    for pass_entries in passes:
        # This list is intentionally mutable: after several genuine timeouts on
        # one provider, the next different provider is promoted ahead of the
        # remaining siblings. Fast 429s still rotate every independent key.
        pass_entries = list(pass_entries)
        for entry_index, mc in enumerate(pass_entries):
            name, provider, llm = mc.name, mc.provider, mc.client

            family_scope = ProviderHealthTracker._family_scope(name)
            timeout_scope = ProviderHealthTracker._timeout_scope(name)
            if family_scope in oversized_scopes:
                logger.info(
                    "⏭️ FAILOVER: skipping %s (same provider/model already rejected oversized request)",
                    name,
                )
                continue
            if timeout_scope in timed_out_families:
                logger.info("⏭️ FAILOVER: skipping %s (request-local key timeout burst)", name)
                continue

            # A repeated timeout is provider/model scoped, not key scoped. This
            # check is deliberately dynamic so later siblings in this same pass
            # are skipped as soon as the strike threshold is reached.
            if _health and _health.in_timeout_cooldown(name):
                logger.info("⏭️  FAILOVER: skipping %s (key timeout cooldown)", name)
                continue

            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                budget_exhausted = True
                break
            if attempt_no >= max_attempts:
                if role is not None:
                    logger.info(
                        "⏭️ FAILOVER: %s role budget exhausted at %d attempts; skipping %s",
                        attempt_role, max_attempts, name,
                    )
                    continue
                # The cap controls repeated sibling-key churn, not diversity.
                # Always preserve a bounded escape to a provider not attempted
                # yet; otherwise five exhausted Gemini keys can prevent a
                # healthy cross-provider fallback from ever being tried.
                if provider in attempted_providers or cross_provider_rescues >= 2:
                    logger.info(
                        "⏭️ FAILOVER: attempt cap reached; skipping sibling %s",
                        name,
                    )
                    continue
                cross_provider_rescues += 1
                logger.info(
                    "🛟 FAILOVER: attempt cap reached; preserving cross-provider rescue via %s",
                    name,
                )

            attempt_no += 1
            attempted_providers.add(provider)

            adaptive = _health.adaptive_timeout(name) if _health else timeout_seconds
            # Previously timeout_seconds was silently ignored whenever health
            # tracking was present. Treat it as the caller's per-attempt cap.
            # A tiny capability-probe latency must not turn a complex DOM/schema
            # request into a false 6-second provider failure.
            effective_floor = (
                min(MODEL_TIMEOUT_FLOOR_SECONDS, timeout_seconds)
                if timeout_seconds > 0
                else MODEL_TIMEOUT_FLOOR_SECONDS
            )
            timeout = max(adaptive, effective_floor)
            if timeout_seconds > 0:
                timeout = min(timeout, timeout_seconds)
            if remaining is not None:
                timeout = min(timeout, remaining)

            # Vision payload injection for multimodal-capable models
            current_messages = messages
            if base64_image and messages:
                supports_vision = any(
                    x in name.lower()
                    for x in [
                        "gpt-4o",
                        "claude-3-5",
                        "gemini",
                        "glm-4v",
                        "vision",
                        "pixtral",
                        "gemma",
                    ]
                )
                from langchain_core.messages import HumanMessage

                last_msg = messages[-1]
                if supports_vision and isinstance(last_msg, HumanMessage):
                    multimodal_content = [
                        {"type": "text", "text": str(last_msg.content)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ]
                    current_messages = messages[:-1] + [HumanMessage(content=multimodal_content)]

            # A fallback model must understand that it is inheriting the SAME
            # request and runtime state. The failed inference attempt returned no
            # usable action and therefore caused no external/browser side effect.
            # Reinforce this at the system level without relying on provider
            # memory (each provider sees a fresh stateless request).
            if attempt_no > 1 and current_messages:
                from langchain_core.messages import SystemMessage
                handoff_note = (
                    "═══ FAILOVER CONTINUATION — AUTHORITATIVE ═══\n"
                    "A previous model attempt failed before returning a usable response. "
                    "It performed NO browser/external action. Continue the exact current "
                    "request from the runtime state, page snapshot, selected markers, "
                    "profile, and execution ledger already supplied. Do not restart, "
                    "repeat completed actions, reuse element IDs from older pages, or "
                    "reinterpret the automation action-turn counter as task progress."
                )
                first = current_messages[0]
                if isinstance(first, SystemMessage):
                    current_messages = [
                        SystemMessage(content=f"{first.content}\n\n{handoff_note}"),
                        *current_messages[1:],
                    ]
                else:
                    current_messages = [SystemMessage(content=handoff_note), *current_messages]

            # Blacklisted (provider, model, schema) combos go straight to JSON mode
            use_json_mode = bool(
                schema is not None and _health and _health.is_schema_blacklisted(name, schema_name)
            )

            t0 = time.monotonic()
            logger.info(
                "🤖 MODEL ATTEMPT START: role=%s model=%s provider=%s attempt=%d/%d timeout=%.1fs remaining=%s input_tokens≈%d messages=%d multimodal=%s json_mode=%s",
                "vision" if base64_image else "text", name, provider,
                attempt_no, total, timeout,
                f"{remaining:.1f}s" if remaining is not None else "unbounded",
                estimated_input_tokens, len(current_messages), bool(base64_image),
                use_json_mode,
            )
            try:
                if _health:
                    if name.startswith("gemini-"):
                        local_usage = _health.quota_snapshot(name)
                        logger.info(
                            "🧮 Gemini project slot %s before call: %d RPM, ~%d TPM, %d RPD locally observed",
                            name.rsplit(":", 1)[-1],
                            local_usage["rpm"],
                            local_usage["tpm"],
                            local_usage["rpd"],
                        )
                    _health.record_attempt(name, estimated_input_tokens)
                if schema is None:
                    response = await _hard_timeout(llm.ainvoke(current_messages), timeout)
                elif use_json_mode:
                    response = await _invoke_json_mode(llm, current_messages, schema, timeout)
                else:
                    structured = llm.with_structured_output(schema)
                    response = await _hard_timeout(structured.ainvoke(current_messages), timeout)

                elapsed = time.monotonic() - t0
                if _health:
                    _health.record_success(name, latency=elapsed)
                    _health.set_preferred_for_role(attempt_role, name)
                if breaker:
                    breaker.record_success()
                attempt_event(mc, attempt_no, t0, classification="SUCCESS", ultimately_used=True)
                logger.info(
                    "🤖 MODEL ATTEMPT SUCCESS: role=%s model=%s elapsed=%.1fs",
                    "vision" if base64_image else "text", name,
                    time.monotonic() - t0,
                )
                return response, name

            except asyncio.TimeoutError:
                last_error = f"{name} timed out (>{timeout:.0f}s)"
                timeout_meta = {"exception_class": "TimeoutError", "diagnostic": last_error}
                attempt_event(mc, attempt_no, t0, classification="TIMEOUT", metadata=timeout_meta)
                if _health:
                    _health.record_failure(name, latency=timeout, failure_class="TIMEOUT")
                    _health.clear_preferred_for_role(attempt_role, name)
                    _health.record_timeout(
                        name,
                        timeout_cooldown_seconds,
                        sibling_threshold=timeout_sibling_threshold,
                    )
                    _health.start_timeout_backoff(name)
                timeout_strikes[timeout_scope] = timeout_strikes.get(timeout_scope, 0) + 1
                provider_timeout_strikes[provider] = provider_timeout_strikes.get(provider, 0) + 1
                if timeout_strikes[timeout_scope] >= max(1, timeout_sibling_threshold):
                    timed_out_families.add(timeout_scope)
                provider_timeout_burst = (
                    VISION_PROVIDER_TIMEOUT_BURST
                    if base64_image
                    else _int_env("MODEL_PROVIDER_TIMEOUT_BURST", 3, minimum=1)
                )
                if provider_timeout_strikes[provider] >= provider_timeout_burst:
                    for later_index in range(entry_index + 1, len(pass_entries)):
                        if pass_entries[later_index].provider != provider:
                            promoted = pass_entries.pop(later_index)
                            pass_entries.insert(entry_index + 1, promoted)
                            logger.info(
                                "🛟 FAILOVER: %d %s timeouts; promoting cross-provider rescue %s",
                                provider_timeout_strikes[provider], provider, promoted.name,
                            )
                            break
                logger.warning(
                    "⚠️  FAILOVER [%d/%d]: role=%s model=%s TIMED OUT (>%.0fs adaptive); elapsed=%.1fs remaining=%s",
                    attempt_no, total, "vision" if base64_image else "text", name,
                    timeout, time.monotonic() - t0,
                    f"{max(0.0, deadline - time.monotonic()):.1f}s" if deadline is not None else "unbounded",
                )

            except Exception as exc:
                classification, metadata = _classify_provider_error(exc, schema=schema)
                err = metadata["diagnostic"] or str(exc)
                last_error = f"{name} — {err[:150]}"
                err_l = err.lower()
                hard_request_too_large = any(
                    marker in err_l
                    for marker in (
                        "request too large", "context length exceeded",
                        "maximum context length", "prompt is too long", "error code: 413",
                    )
                )
                throughput_quota = any(
                    marker in err_l
                    for marker in ("tokens per minute", "tpm:", "tpm limit")
                )
                # TPM is quota pressure, not proof that this prompt can never
                # fit. Treat it as a rate limit on the affected instance so a
                # different key/provider or a later compact request can serve.
                request_too_large = hard_request_too_large
                is_rl = not request_too_large and classification in {"RATE_LIMIT", "QUOTA"}
                is_schema_err = classification == "SCHEMA_INCOMPATIBILITY"
                is_structured_parse_err = classification in {
                    "MALFORMED_STRUCTURED_OUTPUT", "EMPTY_STRUCTURED_OUTPUT"
                }

                if is_structured_parse_err and schema is not None:
                    attempt_event(mc, attempt_no, t0, classification=classification,
                                  metadata=metadata, repair_attempted=True,
                                  repair_type="same_model_json")
                    if _health:
                        _health.record_malformed_output(name, classification)
                        _health.force_json_mode(name, schema_name)
                    logger.warning("⚠️ FAILOVER [%d/%d]: %s returned unusable structured content; same-model repair",
                                   attempt_no, total, name)
                    try:
                        t1 = time.monotonic()
                        response = await _invoke_json_mode(llm, current_messages, schema, timeout)
                        if _health:
                            _health.record_structured_repair(name, True)
                            _health.record_success(name, latency=time.monotonic() - t1)
                        if breaker:
                            breaker.record_success()
                        return response, name
                    except Exception as repair_exc:  # noqa: BLE001
                        if _health:
                            _health.record_structured_repair(name, False)
                        repair_class, repair_meta = _classify_provider_error(repair_exc, schema=schema)
                        attempt_event(mc, attempt_no, t1, classification=repair_class,
                                      metadata=repair_meta, repair_attempted=True,
                                      repair_type="same_model_json")
                        last_error = f"{name} JSON repair — {repair_meta['diagnostic'][:120]}"
                        logger.warning("⚠️ JSON repair failed for %s: %s", name, repair_meta["diagnostic"][:180])
                        continue

                if is_rl:
                    # Per-instance cooldown — the next loop iteration is the
                    # SAME model on the next key/provider. No sleep needed.
                    retry_after = _retry_after_seconds(err)
                    cooldown = max(
                        20.0 + random.uniform(0, 10.0),
                        retry_after or 0.0,
                    )
                    if _health:
                        _health.record_quota_failure(name, err, cooldown)
                        _health.clear_preferred_for_role(attempt_role, name)
                    logger.warning(
                        "⚠️  FAILOVER [%d/%d]: %s rate-limited for %.0fs → %s",
                        attempt_no,
                        total,
                        name,
                        cooldown,
                        _compact_provider_error(err),
                    )
                    continue

                if request_too_large:
                    # Retrying sibling keys for the same provider/model cannot
                    # fix a deterministic prompt-size/TPM rejection. Skip that
                    # family immediately and let a different model handle the
                    # compacted/fallback request within the same total budget.
                    oversized_scopes.add(family_scope)
                    if _health:
                        # 413 is a property of this request's size/quota, not
                        # evidence that the model endpoint is chronically bad.
                        _health.record_failure(
                            name, error_msg=err, reliability_failure=False,
                            failure_class=classification,
                        )
                    logger.warning(
                        "⏭️ FAILOVER: %s rejected the request as oversized; suppressing sibling keys",
                        name,
                    )
                    # Give the same model one compacted attempt before moving
                    # to a weaker model/provider. This is especially important
                    # for provider TPM limits, where changing API keys cannot
                    # make the same oversized prompt fit.
                    compacted_messages = _compact_messages_for_retry(current_messages)
                    compact_timeout = min(
                        timeout,
                        max(0.1, deadline - time.monotonic())
                        if deadline is not None else timeout,
                    )
                    try:
                        if schema is None:
                            compact_response = await _hard_timeout(
                                llm.ainvoke(compacted_messages), compact_timeout
                            )
                        else:
                            compact_response = await _hard_timeout(
                                llm.with_structured_output(schema).ainvoke(compacted_messages),
                                compact_timeout,
                            )
                        if _health:
                            _health.record_success(name, latency=time.monotonic() - t0)
                        if breaker:
                            breaker.record_success()
                        logger.info("✅ FAILOVER: compacted request succeeded on %s", name)
                        return compact_response, name
                    except Exception as compact_exc:  # noqa: BLE001
                        logger.warning(
                            "⚠️ FAILOVER: compacted retry failed for %s: %s",
                            name, str(compact_exc)[:160],
                        )
                    continue

                if is_schema_err and schema is not None:
                    # Register blacklist, then rescue THIS model via JSON mode
                    if _health:
                        _health.record_failure(name, error_msg=err, failure_class=classification)
                        _health.force_json_mode(name, schema_name)
                    attempt_event(mc, attempt_no, t0, classification=classification,
                                  metadata=metadata, repair_attempted=True, repair_type="json_mode")
                    logger.warning(
                        "⚠️  FAILOVER [%d/%d]: %s structured output failed → JSON-mode rescue",
                        attempt_no,
                        total,
                        name,
                    )
                    try:
                        t1 = time.monotonic()
                        response = await _invoke_json_mode(llm, current_messages, schema, timeout)
                        if _health:
                            _health.record_success(name, latency=time.monotonic() - t1)
                        if breaker:
                            breaker.record_success()
                        return response, name
                    except Exception as json_exc:  # noqa: BLE001
                        last_error = f"{name} JSON-mode — {str(json_exc)[:120]}"
                        logger.warning("⚠️  JSON-mode rescue failed for %s: %s", name, str(json_exc)[:150])
                        continue

                if _health:
                    _health.record_failure(name, failure_class=classification)
                    if classification in {"TRANSPORT_ERROR", "HTTP_ERROR", "UNKNOWN", "TIMEOUT", "QUOTA", "RATE_LIMIT"}:
                        _health.clear_preferred_for_role(attempt_role, name)
                attempt_event(mc, attempt_no, t0, classification=classification, metadata=metadata)
                logger.warning(
                    "⚠️  FAILOVER [%d/%d]: role=%s model=%s elapsed=%.1fs — %s",
                    attempt_no, total, "vision" if base64_image else "text", name,
                    time.monotonic() - t0, err[:240],
                )

        if budget_exhausted:
            break

    if breaker:
        # One logical model request failed. Individual provider/key failures are
        # expected failover events and must not prematurely trip the global
        # application breaker before another provider can answer.
        breaker.record_failure()
    if budget_exhausted:
        logger.warning(
            "🛑 FAILOVER STOP: role=%s attempted=%d/%d because total budget %.1fs was exhausted; last=%s",
            "vision" if base64_image else "text", attempt_no, total,
            total_timeout_seconds or 0.0, last_error[:240],
        )
        raise RuntimeError(
            f"Failover time budget exhausted after {attempt_no}/{total} attempts. Last: {last_error}"
        )
    raise RuntimeError(f"ALL {total} models failed. Last: {last_error}")
