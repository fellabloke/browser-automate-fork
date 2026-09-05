"""Provider health, cooldown, probe, and persistence state for model clients.

This module owns the existing health state machine without changing its
public behavior or persistence format.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .schemas import ModelClient

try:
    from app.logger import get_logger

    logger = get_logger("model_registry")
except ImportError:
    logger = logging.getLogger("model_registry")


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


def _gemini_project_limit(instance_name: str, metric: str) -> int:
    singular = f"GEMINI_PROJECT_{metric.upper()}_LIMIT"
    plural = singular + "S"
    values = [value.strip() for value in re.split(r"[,;]", os.getenv(plural, ""))]
    try:
        index = int(instance_name.rsplit(":", 1)[-1])
        if index < len(values) and values[index]:
            return max(0, int(values[index]))
    except (TypeError, ValueError):
        pass
    return _int_env(singular, 0)


def normalize_model_id(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].strip().lower()


MODEL_TIMEOUT_RETRY_COOLDOWN_SECONDS = _float_env(
    "MODEL_TIMEOUT_RETRY_COOLDOWN_SECONDS", 60.0, minimum=1.0
)
MODEL_TIMEOUT_RETRY_COOLDOWN_MAX_SECONDS = _float_env(
    "MODEL_TIMEOUT_RETRY_COOLDOWN_MAX_SECONDS", 60.0, minimum=1.0
)


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


