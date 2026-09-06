"""Ordinary model invocation, retry/failover, and structured recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import deque
from datetime import datetime
from typing import Any

from .health import ProviderHealthTracker, _float_env, _int_env
from .routing import order_failover_chain
from .schemas import ModelClient

try:
    from agent_first_browse.logging import get_logger

    logger = get_logger("model_registry")
except ImportError:
    logger = logging.getLogger("model_registry")


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


MODEL_TIMEOUT_FLOOR_SECONDS = _float_env("MODEL_TIMEOUT_FLOOR_SECONDS", 5.0, minimum=1.0)
MODEL_FAILOVER_BUDGET_SECONDS = _float_env("MODEL_FAILOVER_BUDGET_SECONDS", 15.0, minimum=1.0)
MODEL_FAILOVER_MAX_ATTEMPTS = _int_env("MODEL_FAILOVER_MAX_ATTEMPTS", 5, minimum=1)
VISION_PROVIDER_TIMEOUT_BURST = _int_env("VISION_PROVIDER_TIMEOUT_BURST", 1, minimum=1)
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


def _compact_provider_error(error: str, limit: int = 280) -> str:
    """Keep diagnostics useful while ensuring credentials never reach logs."""
    compact = re.sub(r"\s+", " ", error).strip()
    compact = re.sub(
        r"\b(?:gsk_|nvapi-|csk-|cfut_|sk-|AIza)[A-Za-z0-9_.-]+",
        "<redacted-key>",
        compact,
    )
    return compact[:limit] or "provider returned HTTP 429"


def _extract_provider_error_metadata(exc: BaseException, response: Any = None) -> dict[str, Any]:
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


def classify_provider_error(
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
        category = "SCHEMA_INCOMPATIBILITY" if schema is not None and any(
            x in text for x in ("schema", "response_format", "additionalproperties")
        ) else "HTTP_ERROR"
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


def _as_model_clients(chain: list) -> list[ModelClient]:
    entries: list[ModelClient] = []
    for item in chain:
        if isinstance(item, ModelClient):
            entries.append(item)
        else:
            name = getattr(item, "model_name", getattr(item, "model", str(item)))
            entries.append(ModelClient(name=str(name), client=item, provider="unknown", pipeline="text"))
    return entries


def extract_text(raw_response: Any) -> str:
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


def estimate_input_tokens(messages: list, schema: type | None = None) -> int:
    characters = 0
    for message in messages or []:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
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


def extract_json_payload(text: str) -> dict:
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


def retry_after_seconds(error: str) -> float | None:
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
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


async def hard_timeout(awaitable: Any, timeout: float) -> Any:
    task = asyncio.ensure_future(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=max(0.001, float(timeout)))
    if task not in done:
        task.cancel()
        task.add_done_callback(_drain_cancelled_task)
        raise asyncio.TimeoutError
    if task.cancelled():
        raise asyncio.TimeoutError
    return task.result()


async def invoke_json_mode(llm, messages: list, schema: type, timeout: float) -> Any:
    """Invoke one client in plain JSON mode and validate the structured result."""
    from langchain_core.messages import HumanMessage

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    instruction = HumanMessage(
        content=(
            "Respond ONLY with a single JSON object that matches this JSON Schema. "
            "No markdown fences, no commentary, no extra keys.\n"
            f"SCHEMA: {schema_json}"
        )
    )
    raw = await hard_timeout(llm.ainvoke(list(messages) + [instruction]), timeout)
    return schema.model_validate(extract_json_payload(extract_text(raw)))


def compact_messages_for_retry(messages: list, max_chars: int = 9000) -> list:
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


class CircuitBreaker:
    """Three-state circuit breaker used by ordinary inference."""

    def __init__(self, window_size=50, min_calls=10, failure_rate_threshold=0.5,
                 open_wait_secs=30.0, half_open_probes=3, max_requests=80, max_failures=15):
        self._window = deque(maxlen=window_size)
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
    def state(self):
        return self._state

    @property
    def tripped(self):
        if self._state == "open":
            return time.monotonic() - self._opened_at < self._open_wait_secs
        return False

    async def allow(self):
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
            if self._probes_done < self._max_probes:
                self._probes_done += 1
                return True
            return False

    async def record(self, is_failure: bool):
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

    def record_success(self):
        self._total_requests += 1
        self._window.append(False)

    def record_failure(self):
        self._total_requests += 1
        self._total_failures += 1
        self._window.append(True)
        if len(self._window) >= self._min_calls:
            rate = sum(self._window) / len(self._window)
            if rate >= self._failure_rate_threshold:
                self._trip(f"Failure rate {rate:.0%} (sync)")

    def _trip(self, reason):
        self._state = "open"
        self._opened_at = time.monotonic()
        logger.warning("🔌 Circuit breaker → OPEN: %s", reason)

    def _reset(self):
        self._state = "closed"
        self._window.clear()
        self._probes_done = 0
        logger.info("✅ Circuit breaker → CLOSED (recovered)")

    @property
    def reason(self):
        if self._state == "open":
            remaining = max(0, self._open_wait_secs - (time.monotonic() - self._opened_at))
            return f"OPEN (recovery in {remaining:.0f}s)"
        return ""

    def status_line(self):
        fails = sum(self._window)
        return f"[CB:{self._state} {fails}/{len(self._window)} fails, {self._total_requests} total]"


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
    health_tracker: ProviderHealthTracker | None = None,
    base64_image: str | None = None,
    role: str | None = None,
    max_attempts: int | None = None,
) -> tuple[Any, str]:
    """Invoke candidates in order with the existing bounded recovery policy."""
    _health = health or health_tracker
    if total_timeout_seconds is None:
        total_timeout_seconds = MODEL_FAILOVER_BUDGET_SECONDS
    requested_max_attempts = max_attempts
    if breaker and breaker.tripped:
        raise RuntimeError(f"Circuit breaker tripped: {breaker.reason}")

    entries = _as_model_clients(chain)
    if _health:
        benched = [
            mc for mc in entries
            if (mc.credential_id or not _health.has_persistence)
            and _health.is_chronically_unreliable(mc.name)
        ]
        entries = [
            mc for mc in entries
            if not ((mc.credential_id or not _health.has_persistence)
                    and _health.is_chronically_unreliable(mc.name))
        ]
        if benched:
            logger.warning("🪑 FAILOVER: benched %d chronically unreliable model instance(s)", len(benched))
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
    deadline = time.monotonic() + total_timeout_seconds if total_timeout_seconds and total_timeout_seconds > 0 else None

    if _health:
        quota_guarded = {mc.name for mc in ordered if _health.in_quota_guard(mc.name, critical=mc.critical)}
        hot = [mc for mc in ordered if mc.name not in quota_guarded and not _health.in_cooldown(mc.name, critical=mc.critical)]
        cooled = [mc for mc in ordered if mc.name not in quota_guarded and _health.in_cooldown(mc.name, critical=mc.critical)]
        if quota_guarded:
            logger.info("🧮 QUOTA GUARD: parked %d key/model instance(s) at their configured free-limit threshold", len(quota_guarded))
    else:
        hot, cooled = ordered, []
    passes = [hot] if hot else []
    if _env_flag("MODEL_ALLOW_COOLED_LAST_RESORT", False) and cooled:
        passes.append(cooled)
    if not passes:
        raise RuntimeError("All reliable model instances are cooling down or quota-guarded")

    schema_name = getattr(schema, "__name__", "__structured_output__") if schema is not None else ""
    last_error = ""
    attempt_no = 0
    attempted_providers: set[str] = set()
    cross_provider_rescues = 0
    total = len(entries)
    budget_exhausted = False
    oversized_scopes: set[str] = set()
    timed_out_families: set[str] = set()
    timeout_strikes: dict[str, int] = {}
    provider_timeout_strikes: dict[str, int] = {}
    estimated_input_tokens = estimate_input_tokens(messages, schema)

    def attempt_event(mc, attempt, started, *, classification, metadata=None,
                      repair_attempted=False, repair_type="", ultimately_used=False):
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
        pass_entries = list(pass_entries)
        for entry_index, mc in enumerate(pass_entries):
            name, provider, llm = mc.name, mc.provider, mc.client
            family_scope = ProviderHealthTracker._family_scope(name)
            timeout_scope = ProviderHealthTracker._timeout_scope(name)
            if family_scope in oversized_scopes or timeout_scope in timed_out_families:
                continue
            if _health and _health.in_timeout_cooldown(name):
                continue
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                budget_exhausted = True
                break
            if attempt_no >= max_attempts:
                if role is not None:
                    continue
                if provider in attempted_providers or cross_provider_rescues >= 2:
                    continue
                cross_provider_rescues += 1
            attempt_no += 1
            attempted_providers.add(provider)

            adaptive = _health.adaptive_timeout(name) if _health else timeout_seconds
            effective_floor = min(MODEL_TIMEOUT_FLOOR_SECONDS, timeout_seconds) if timeout_seconds > 0 else MODEL_TIMEOUT_FLOOR_SECONDS
            timeout = max(adaptive, effective_floor)
            if timeout_seconds > 0:
                timeout = min(timeout, timeout_seconds)
            if remaining is not None:
                timeout = min(timeout, remaining)

            current_messages = messages
            if base64_image and messages:
                supports_vision = any(x in name.lower() for x in ("gpt-4o", "claude-3-5", "gemini", "glm-4v", "vision", "pixtral", "gemma"))
                from langchain_core.messages import HumanMessage
                last_msg = messages[-1]
                if supports_vision and isinstance(last_msg, HumanMessage):
                    multimodal_content = [
                        {"type": "text", "text": str(last_msg.content)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ]
                    current_messages = messages[:-1] + [HumanMessage(content=multimodal_content)]
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
                    current_messages = [SystemMessage(content=f"{first.content}\n\n{handoff_note}"), *current_messages[1:]]
                else:
                    current_messages = [SystemMessage(content=handoff_note), *current_messages]

            use_json_mode = bool(schema is not None and _health and _health.is_schema_blacklisted(name, schema_name))
            t0 = time.monotonic()
            try:
                if _health:
                    if name.startswith("gemini-"):
                        local_usage = _health.quota_snapshot(name)
                        logger.info("🧮 Gemini project slot %s before call: %d RPM, ~%d TPM, %d RPD locally observed", name.rsplit(":", 1)[-1], local_usage["rpm"], local_usage["tpm"], local_usage["rpd"])
                    _health.record_attempt(name, estimated_input_tokens)
                if schema is None:
                    response = await hard_timeout(llm.ainvoke(current_messages), timeout)
                elif use_json_mode:
                    response = await invoke_json_mode(llm, current_messages, schema, timeout)
                else:
                    response = await hard_timeout(llm.with_structured_output(schema).ainvoke(current_messages), timeout)
                elapsed = time.monotonic() - t0
                if _health:
                    _health.record_success(name, latency=elapsed)
                    _health.set_preferred_for_role(attempt_role, name)
                if breaker:
                    breaker.record_success()
                attempt_event(mc, attempt_no, t0, classification="SUCCESS", ultimately_used=True)
                return response, name

            except asyncio.TimeoutError:
                last_error = f"{name} timed out (>{timeout:.0f}s)"
                attempt_event(mc, attempt_no, t0, classification="TIMEOUT", metadata={"exception_class": "TimeoutError", "diagnostic": last_error})
                if _health:
                    _health.record_failure(name, latency=timeout, failure_class="TIMEOUT")
                    _health.clear_preferred_for_role(attempt_role, name)
                    _health.record_timeout(name, timeout_cooldown_seconds, sibling_threshold=timeout_sibling_threshold)
                    _health.start_timeout_backoff(name)
                timeout_strikes[timeout_scope] = timeout_strikes.get(timeout_scope, 0) + 1
                provider_timeout_strikes[provider] = provider_timeout_strikes.get(provider, 0) + 1
                if timeout_strikes[timeout_scope] >= max(1, timeout_sibling_threshold):
                    timed_out_families.add(timeout_scope)
                provider_timeout_burst = VISION_PROVIDER_TIMEOUT_BURST if base64_image else _int_env("MODEL_PROVIDER_TIMEOUT_BURST", 3, minimum=1)
                if provider_timeout_strikes[provider] >= provider_timeout_burst:
                    for later_index in range(entry_index + 1, len(pass_entries)):
                        if pass_entries[later_index].provider != provider:
                            pass_entries.insert(entry_index + 1, pass_entries.pop(later_index))
                            break

            except Exception as exc:
                classification, metadata = classify_provider_error(exc, schema=schema)
                err = metadata["diagnostic"] or str(exc)
                last_error = f"{name} — {err[:150]}"
                err_l = err.lower()
                hard_request_too_large = any(marker in err_l for marker in ("request too large", "context length exceeded", "maximum context length", "prompt is too long", "error code: 413"))
                request_too_large = hard_request_too_large
                is_rl = not request_too_large and classification in {"RATE_LIMIT", "QUOTA"}
                is_schema_err = classification == "SCHEMA_INCOMPATIBILITY"
                is_structured_parse_err = classification in {"MALFORMED_STRUCTURED_OUTPUT", "EMPTY_STRUCTURED_OUTPUT"}

                if is_structured_parse_err and schema is not None:
                    attempt_event(mc, attempt_no, t0, classification=classification, metadata=metadata, repair_attempted=True, repair_type="same_model_json")
                    if _health:
                        _health.record_malformed_output(name, classification)
                        _health.force_json_mode(name, schema_name)
                    try:
                        t1 = time.monotonic()
                        response = await invoke_json_mode(llm, current_messages, schema, timeout)
                        if _health:
                            _health.record_structured_repair(name, True)
                            _health.record_success(name, latency=time.monotonic() - t1)
                        if breaker:
                            breaker.record_success()
                        return response, name
                    except Exception as repair_exc:
                        if _health:
                            _health.record_structured_repair(name, False)
                        repair_class, repair_meta = classify_provider_error(repair_exc, schema=schema)
                        attempt_event(mc, attempt_no, t1, classification=repair_class, metadata=repair_meta, repair_attempted=True, repair_type="same_model_json")
                        last_error = f"{name} JSON repair — {repair_meta['diagnostic'][:120]}"
                        continue

                if is_rl:
                    cooldown = max(20.0 + random.uniform(0, 10.0), retry_after_seconds(err) or 0.0)
                    if _health:
                        _health.record_quota_failure(name, err, cooldown)
                        _health.clear_preferred_for_role(attempt_role, name)
                    continue

                if request_too_large:
                    oversized_scopes.add(family_scope)
                    if _health:
                        _health.record_failure(name, error_msg=err, reliability_failure=False, failure_class=classification)
                    compacted_messages = compact_messages_for_retry(current_messages)
                    compact_timeout = min(timeout, max(0.1, deadline - time.monotonic()) if deadline is not None else timeout)
                    try:
                        if schema is None:
                            compact_response = await hard_timeout(llm.ainvoke(compacted_messages), compact_timeout)
                        else:
                            compact_response = await hard_timeout(llm.with_structured_output(schema).ainvoke(compacted_messages), compact_timeout)
                        if _health:
                            _health.record_success(name, latency=time.monotonic() - t0)
                        if breaker:
                            breaker.record_success()
                        return compact_response, name
                    except Exception:
                        pass
                    continue

                if is_schema_err and schema is not None:
                    if _health:
                        _health.record_failure(name, error_msg=err, failure_class=classification)
                        _health.force_json_mode(name, schema_name)
                    attempt_event(mc, attempt_no, t0, classification=classification, metadata=metadata, repair_attempted=True, repair_type="json_mode")
                    try:
                        response = await invoke_json_mode(llm, current_messages, schema, timeout)
                        if _health:
                            _health.record_success(name, latency=time.monotonic() - t0)
                        if breaker:
                            breaker.record_success()
                        return response, name
                    except Exception as json_exc:
                        last_error = f"{name} JSON-mode — {str(json_exc)[:120]}"
                        continue

                if _health:
                    _health.record_failure(name, failure_class=classification)
                    if classification in {"TRANSPORT_ERROR", "HTTP_ERROR", "UNKNOWN", "TIMEOUT", "QUOTA", "RATE_LIMIT"}:
                        _health.clear_preferred_for_role(attempt_role, name)
                attempt_event(mc, attempt_no, t0, classification=classification, metadata=metadata)

        if budget_exhausted:
            break

    if breaker:
        breaker.record_failure()
    if budget_exhausted:
        raise RuntimeError(f"Failover time budget exhausted after {attempt_no}/{total} attempts. Last: {last_error}")
    raise RuntimeError(f"ALL {total} models failed. Last: {last_error}")
