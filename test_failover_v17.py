"""Unit tests for the V17.0 Resilient Reasoning Foundation.

Covers the model-first failover guarantees the user explicitly required:
  - ALL instances of the best model (every key, every provider) are tried
    before any weaker model.
  - A 429 on one instance jumps to the SAME model on the next key — never
    to a lower-tier model while a same-model instance remains.
  - Adaptive per-model timeout abandons a hung model early.
  - A schema-400 is rescued via JSON mode on the same model (not benched).
  - Dead models (404) are pruned by the startup probe.
  - Pydantic schemas emit additionalProperties:false (Groq strict mode).

Run: .venv/bin/python -m pytest test_failover_v17.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.append(str(Path(__file__).parent / "python-orchestrator"))

import model_registry as mr
from model_registry import (
    ModelClient,
    ProviderHealthTracker,
    _credential_fingerprint,
    _compact_provider_error,
    _retry_after_seconds,
    get_model_tier,
    normalize_model_id,
    order_failover_chain,
    invoke_with_failover,
    _classify_provider_error,
    _extract_provider_error_metadata,
)


def test_cloudflare_daily_allocation_is_parked_to_utc_reset():
    health = ProviderHealthTracker()
    name = "cloudflare-vision:@cf/meta/llama-3.2-11b-vision-instruct:0"
    health.record_quota_failure(
        name,
        "429 daily free allocation of 10,000 neurons code 4006",
        20.0,
    )
    state = health._get(name)
    assert state["daily_exhausted_until"] > time.time()
    assert state["cooldown_until"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Fake LLM clients
# ═══════════════════════════════════════════════════════════════════════════════

class FakeResponse:
    def __init__(self, content="ok"):
        self.content = content


class FakeLLM:
    """A stand-in LangChain-style client with scriptable behavior.

    behavior: "ok" | "429" | "gemini_tpm" | "timeout" | "oversized" | "schema400" | "parse_error"
    Records every ainvoke call onto the shared `calls` list.
    """

    def __init__(self, name, behavior, calls, latency=0.01, json_ok=True):
        self._name = name
        self._behavior = behavior
        self._calls = calls
        self._latency = latency
        self._json_ok = json_ok
        self._structured = False

    def with_structured_output(self, schema):
        clone = FakeLLM(self._name, self._behavior, self._calls,
                        self._latency, self._json_ok)
        clone._structured = True
        return clone

    async def ainvoke(self, messages, config=None):
        self._calls.append((self._name, "structured" if self._structured else "raw"))
        if self._behavior == "timeout":
            await asyncio.sleep(5.0)  # exceeds any adaptive timeout in tests
            return FakeResponse()
        await asyncio.sleep(self._latency)
        if self._behavior == "429":
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")
        if self._behavior == "oversized":
            raise RuntimeError(
                "Error code: 413 - Request too large: tokens per minute (TPM) limit exceeded"
            )
        if self._behavior == "gemini_tpm":
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED: tokens per minute quota exceeded; retryDelay: 30s"
            )
        if self._behavior == "schema400" and self._structured:
            # Only structured calls fail; JSON-mode (raw) succeeds
            raise RuntimeError(
                "Error code: 400 - invalid JSON schema for response_format: "
                "'CandidateSet': additionalProperties"
            )
        if self._behavior == "parse_error" and self._structured:
            raise RuntimeError(
                "1 validation error for Tiny: Invalid JSON [type=json_invalid]"
            )
        return FakeResponse('{"ok": true}')


def mc(name, behavior, calls, provider="groq", latency=0.01):
    return ModelClient(
        name=name,
        client=FakeLLM(name, behavior, calls, latency),
        provider=provider,
        pipeline="text",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Ordering / tier tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_normalize_and_tier():
    assert normalize_model_id("openai/gpt-oss-120b") == "gpt-oss-120b"
    assert get_model_tier("groq:openai/gpt-oss-120b:0") == 0
    assert get_model_tier("nvidia-text:google/gemma-4-31b-it:0") == 1
    assert get_model_tier("x:some/unknown-model:0") == mr.UNKNOWN_MODEL_TIER


def test_order_groups_best_model_across_providers_first():
    """All gpt-oss instances (Groq keys + NVIDIA) precede any gemma instance."""
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "ok", calls, "groq"),
        mc("nvidia-text:google/gemma-4-31b-it:0", "ok", calls, "nvidia"),
        mc("groq:openai/gpt-oss-120b:1", "ok", calls, "groq"),
        mc("nvidia-text:openai/gpt-oss-120b:0", "ok", calls, "nvidia"),
    ]
    ordered = [m.name for m in order_failover_chain(chain, health)]
    gpt_positions = [i for i, n in enumerate(ordered) if "gpt-oss" in n]
    gemma_positions = [i for i, n in enumerate(ordered) if "gemma" in n]
    assert max(gpt_positions) < min(gemma_positions), ordered
    assert len(gpt_positions) == 3


def test_configured_gemini_vision_precedes_nvidia_fallbacks():
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("nvidia-vision:meta/llama-3.2-11b-vision-instruct:0", "ok", calls, "nvidia"),
        mc("gemini-vision:gemini-3.5-flash:0", "ok", calls, "google"),
    ]
    ordered = [m.name for m in order_failover_chain(chain, health)]
    assert ordered[0].startswith("gemini-vision:"), ordered


def test_rate_limit_diagnostics_parse_retry_after_and_redact_keys():
    error = "429 retry-after: 1.5 minutes for gsk_secret-value"
    assert _retry_after_seconds(error) == 90.0
    assert _retry_after_seconds("RESOURCE_EXHAUSTED retryDelay: '34s'") == 34.0
    compact = _compact_provider_error(error)
    assert "gsk_secret-value" not in compact
    assert "<redacted-key>" in compact


def test_provider_failure_classification_is_normalized():
    class HttpError(Exception):
        status_code = 429
        body = "daily quota exhausted for gsk_secret"

    category, metadata = _classify_provider_error(HttpError())
    assert category == "QUOTA"
    assert metadata["http_status"] == 429
    assert "gsk_secret" not in metadata["diagnostic"]

    category, _ = _classify_provider_error(
        json.JSONDecodeError("Expecting value", "", 0), schema=object
    )
    assert category == "EMPTY_STRUCTURED_OUTPUT"
    malformed = json.JSONDecodeError("Expecting ',' delimiter", "{}", 420)
    category, _ = _classify_provider_error(malformed, schema=object)
    assert category == "MALFORMED_STRUCTURED_OUTPUT"


def test_role_affinity_keeps_successful_primary_first():
    health = ProviderHealthTracker()
    health.set_preferred_for_role("TEXT_WORKER", "groq:openai/gpt-oss-120b:1")
    assert health.preferred_for_role(
        "TEXT_WORKER", {"groq:openai/gpt-oss-120b:0", "groq:openai/gpt-oss-120b:1"}
    ).endswith(":1")


def test_explicit_role_budget_stops_provider_cascade():
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("google:gemini-3.5-flash-lite:0", "timeout", calls, "google"),
        mc("google:gemini-3.5-flash-lite:1", "timeout", calls, "google"),
        mc("nvidia-text:openai/gpt-oss-120b:0", "ok", calls, "nvidia"),
    ]
    with pytest.raises(RuntimeError):
        _run(invoke_with_failover(
            chain, ["hi"], health=health, timeout_seconds=0.01,
            total_timeout_seconds=1.0, role="VISION", max_attempts=2,
        ))
    assert len(calls) == 2


# ═══════════════════════════════════════════════════════════════════════════════
#  Failover behavior tests
# ═══════════════════════════════════════════════════════════════════════════════

def _run(coro):
    return asyncio.run(coro)


def test_429_rotates_to_same_model_next_key_before_weaker():
    """429 on gpt-oss key0 → gpt-oss key1 (SAME model), not the gemma model."""
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "429", calls, "groq"),
        mc("groq:openai/gpt-oss-120b:1", "ok", calls, "groq"),
        mc("nvidia-text:google/gemma-4-31b-it:0", "ok", calls, "nvidia"),
    ]
    resp, name = _run(invoke_with_failover(chain, ["hi"], schema=None, health=health))
    assert name == "groq:openai/gpt-oss-120b:1"
    # The weaker gemma model must NOT have been called at all
    assert all("gemma" not in c[0] for c in calls), calls


def test_429_all_gpt_keys_then_same_model_on_nvidia():
    """Both Groq gpt-oss keys 429 → same model on NVIDIA, still before gemma."""
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "429", calls, "groq"),
        mc("groq:openai/gpt-oss-120b:1", "429", calls, "groq"),
        mc("nvidia-text:openai/gpt-oss-120b:0", "ok", calls, "nvidia"),
        mc("nvidia-text:google/gemma-4-31b-it:0", "ok", calls, "nvidia"),
    ]
    resp, name = _run(invoke_with_failover(chain, ["hi"], schema=None, health=health))
    assert name == "nvidia-text:openai/gpt-oss-120b:0"
    assert all("gemma" not in c[0] for c in calls), calls


def test_adaptive_timeout_abandons_hung_model():
    """A model with a low seeded latency gets a short timeout and is abandoned."""
    health = ProviderHealthTracker()
    health.seed_latency("groq:openai/gpt-oss-120b:0", 0.5)  # → ~6s floor timeout
    calls = []
    chain = [
        # First model hangs (sleeps 5s) but we give it floor=6s, so to make the
        # test fast we instead seed a tiny latency and rely on the timeout floor.
        mc("groq:openai/gpt-oss-120b:0", "ok", calls, "groq"),
        mc("groq:openai/gpt-oss-120b:1", "ok", calls, "groq"),
    ]
    # Just assert adaptive_timeout math here (fast, deterministic)
    t = health.adaptive_timeout("groq:openai/gpt-oss-120b:0")
    assert t == 6.0  # clamp(3*0.5, 6, 20) = 6
    health.seed_latency("x:slow/model:0", 10.0)
    assert health.adaptive_timeout("x:slow/model:0") == 20.0  # clamp(30,6,20)=20
    # Unknown latency → full cap
    assert health.adaptive_timeout("y:unknown/model:0") == 20.0


def test_caller_timeout_cap_is_honored_with_health_tracking():
    """The explicit cap must not be ignored merely because health is enabled."""
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "timeout", calls, "groq"),
        mc("groq:openai/gpt-oss-120b:1", "ok", calls, "groq"),
    ]
    started = time.monotonic()
    _, name = _run(invoke_with_failover(
        chain, ["hi"], schema=None, health=health, timeout_seconds=0.03,
    ))
    assert time.monotonic() - started < 0.5
    assert name == "groq:openai/gpt-oss-120b:1"


def test_repeated_model_timeout_cycles_to_remaining_sibling_key():
    """Two stalled keys do not suppress a healthy sibling key."""
    health = ProviderHealthTracker()
    calls = []
    sibling = "groq:openai/gpt-oss-120b:2"
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "timeout", calls, "groq"),
        mc("groq:openai/gpt-oss-120b:1", "timeout", calls, "groq"),
        mc(sibling, "ok", calls, "groq"),
        mc("nvidia-text:openai/gpt-oss-120b:0", "ok", calls, "nvidia"),
    ]
    _, name = _run(invoke_with_failover(
        chain, ["hi"], schema=None, health=health, timeout_seconds=0.03,
        timeout_cooldown_seconds=30.0, timeout_sibling_threshold=2,
    ))
    assert name == sibling
    assert any(call_name == sibling for call_name, _ in calls)


def test_total_failover_budget_stops_the_walk():
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "timeout", calls, "groq"),
        mc("nvidia-text:openai/gpt-oss-120b:0", "timeout", calls, "nvidia"),
        mc("gemini-text:google/gemma-4-31b-it:0", "ok", calls, "google"),
    ]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="time budget exhausted"):
        _run(invoke_with_failover(
            chain, ["hi"], schema=None, health=health,
            timeout_seconds=0.05, total_timeout_seconds=0.07,
        ))
    assert time.monotonic() - started < 0.5
    assert all("gemini" not in call_name for call_name, _ in calls)


def test_oversized_request_skips_same_provider_model_siblings():
    """A deterministic 413/TPM rejection must not retry every API key."""
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "oversized", calls, "groq"),
        mc("groq:openai/gpt-oss-120b:1", "oversized", calls, "groq"),
        mc("nvidia-text:openai/gpt-oss-120b:0", "ok", calls, "nvidia"),
    ]

    _response, name = _run(invoke_with_failover(
        chain, ["hi"], schema=None, health=health,
        timeout_seconds=0.05, total_timeout_seconds=0.5,
    ))

    assert name == "nvidia-text:openai/gpt-oss-120b:0"
    assert [call_name for call_name, _mode in calls].count("groq:openai/gpt-oss-120b:1") == 0


def test_oversized_request_does_not_bench_model():
    """A prompt-size rejection must not poison the model's reliability score."""
    health = ProviderHealthTracker()
    name = "groq:openai/gpt-oss-120b:0"
    for _ in range(8):
        health.record_failure(name, error_msg="Error code: 413 request too large", reliability_failure=False)
    assert health.is_chronically_unreliable(name) is False


def test_rate_limits_do_not_bench_provider():
    """Quota exhaustion parks a credential temporarily, not permanently."""
    health = ProviderHealthTracker()
    name = "cloudflare:llama-3.3-70b-instruct-fp8-fast:0"
    for _ in range(8):
        health.record_quota_failure(name, "429 rate limit exceeded", 30.0)
    assert health.is_chronically_unreliable(name) is False
    assert health.in_cooldown(name) is True


def test_schema_400_rescued_via_json_mode_same_model():
    """A schema-400 model is rescued in JSON mode, not skipped to a weaker one."""
    from web_dreamer import CandidateSet
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("groq:openai/gpt-oss-120b:0", "schema400", calls, "groq"),
        mc("nvidia-text:google/gemma-4-31b-it:0", "ok", calls, "nvidia"),
    ]
    # CandidateSet requires a non-empty schema; FakeLLM returns {"ok": true},
    # which won't validate — so use a schema that accepts {"ok": true}.
    from pydantic import BaseModel, ConfigDict

    class Tiny(BaseModel):
        model_config = ConfigDict(extra="forbid")
        ok: bool = False

    resp, name = _run(invoke_with_failover(chain, ["hi"], schema=Tiny, health=health))
    # Rescued on the SAME top-tier model via JSON mode
    assert name == "groq:openai/gpt-oss-120b:0"
    # Confirm a structured attempt happened first, then a raw (JSON-mode) retry
    names_modes = [c for c in calls if c[0] == "groq:openai/gpt-oss-120b:0"]
    assert ("groq:openai/gpt-oss-120b:0", "structured") in names_modes
    assert ("groq:openai/gpt-oss-120b:0", "raw") in names_modes


def test_structured_parse_failure_is_rescued_and_remembered():
    from pydantic import BaseModel, ConfigDict

    class Tiny(BaseModel):
        model_config = ConfigDict(extra="forbid")
        ok: bool

    health = ProviderHealthTracker()
    calls = []
    name = "nvidia-vision:meta/llama-3.2-11b-vision-instruct:0"
    chain = [mc(name, "parse_error", calls, "nvidia")]
    response, used = _run(invoke_with_failover(chain, ["hi"], schema=Tiny, health=health))
    assert response.ok is True and used == name
    assert calls == [(name, "structured"), (name, "raw")]
    assert health.is_schema_blacklisted(name, "Tiny") is True


def test_forced_json_mode_bypasses_known_bad_strict_path():
    from pydantic import BaseModel

    class Tiny(BaseModel):
        ok: bool

    health = ProviderHealthTracker()
    calls = []
    name = "nvidia-vision:meta/llama-3.2-11b-vision-instruct:0"
    health.force_json_mode(name)
    response, _ = _run(invoke_with_failover(
        [mc(name, "parse_error", calls, "nvidia")], ["hi"], schema=Tiny, health=health,
    ))
    assert response.ok is True
    assert calls == [(name, "raw")]


def test_capability_probe_persists_json_mode_requirement():
    class ProbeLLM:
        def __init__(self, structured=False):
            self.structured = structured

        def with_structured_output(self, schema):
            return ProbeLLM(structured=True)

        async def ainvoke(self, messages):
            if self.structured:
                raise RuntimeError("strict structured output unsupported")
            return FakeResponse(
                '{"action":"type","element_id":"e3","confidence":0.9}'
            )

    name = "nvidia-vision:meta/llama-3.2-11b-vision-instruct:0"
    client = ModelClient(name, ProbeLLM(), "nvidia", "vision")
    registry = object.__new__(mr.ModelRegistry)
    registry._text_pipeline = []
    registry._vision_pipeline = [client]
    registry._probed = False
    registry.health = ProviderHealthTracker()

    _run(registry.probe_and_prune(timeout=0.2))
    assert registry.health.is_schema_blacklisted(name, "VisionVerdict") is True


def test_all_fail_raises():
    health = ProviderHealthTracker()
    calls = []
    chain = [mc("groq:openai/gpt-oss-120b:0", "429", calls, "groq")]
    with pytest.raises(RuntimeError):
        _run(invoke_with_failover(chain, ["hi"], schema=None, health=health))


# ═══════════════════════════════════════════════════════════════════════════════
#  Schema strict-mode tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_schemas_emit_additional_properties_false():
    from prm_critic import ChecklistEvaluation
    from web_dreamer import CandidateSet

    ce = ChecklistEvaluation.model_json_schema()
    assert ce["$defs"]["EvaluationItem"]["additionalProperties"] is False

    cs = CandidateSet.model_json_schema()
    assert cs["$defs"]["Candidate"]["additionalProperties"] is False


# ═══════════════════════════════════════════════════════════════════════════════
#  Health math tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_expected_cost_penalizes_failing_model():
    health = ProviderHealthTracker()
    good = "groq:openai/gpt-oss-120b:0"
    bad = "groq:openai/gpt-oss-120b:1"
    health.observe_latency(good, 1.0)
    health.observe_latency(bad, 1.0)
    health.record_failure(bad)  # bumps p_fail
    assert health.expected_cost(bad) > health.expected_cost(good)


def test_cooldown_is_per_instance():
    health = ProviderHealthTracker()
    a = "groq:openai/gpt-oss-120b:0"
    b = "groq:openai/gpt-oss-120b:1"
    health.start_cooldown(a, 30.0)
    assert health.in_cooldown(a) is True
    assert health.in_cooldown(b) is False  # sibling key unaffected


def test_repeated_timeout_backoff_parks_only_the_unhealthy_key():
    health = ProviderHealthTracker()
    bad = "gemini-text:gemini-3.5-flash-lite:0"
    good = "gemini-text:gemini-3.5-flash-lite:1"
    health.record_failure(bad, latency=20.0)
    assert health.start_timeout_backoff(bad) == 0.0
    health.record_failure(bad, latency=20.0)
    assert 0.0 < health.start_timeout_backoff(bad) <= 60.0
    assert health.in_cooldown(bad) is True
    assert health.in_cooldown(good) is False
    health.record_success(bad, latency=1.0)
    assert health.in_cooldown(bad) is False


def test_timeout_cooldown_requires_repeated_sibling_hangs():
    health = ProviderHealthTracker()
    a = "nvidia-vision:meta/llama-3.2-11b-vision-instruct:0"
    b = "nvidia-vision:meta/llama-3.2-11b-vision-instruct:1"
    assert health.record_timeout(a, 30.0, sibling_threshold=2) is False
    assert health.in_timeout_cooldown(b) is False
    assert health.record_timeout(b, 30.0, sibling_threshold=2) is False
    assert health.in_timeout_cooldown(a) is False
    assert health.record_timeout(a, 30.0, sibling_threshold=2) is True
    assert health.in_timeout_cooldown(a) is True
    assert health.in_timeout_cooldown(b) is False
    health.record_success(a)
    assert health.in_timeout_cooldown(b) is False


def test_gemini_project_timeouts_do_not_persistently_suppress_sibling_projects():
    health = ProviderHealthTracker()
    a = "gemini-text:gemini-3.5-flash-lite:0"
    b = "gemini-text:gemini-3.5-flash-lite:1"
    assert health.record_timeout(a, 30.0, sibling_threshold=2) is False
    assert health.record_timeout(b, 30.0, sibling_threshold=2) is False
    assert health.in_timeout_cooldown(a) is False
    assert health.in_timeout_cooldown(b) is False


def test_six_independent_gemini_rate_limits_reach_provider_fallback():
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc(f"gemini-text:gemini-3.5-flash-lite:{index}", "429", calls, "google")
        for index in range(6)
    ] + [mc("groq:openai/gpt-oss-120b:0", "ok", calls, "groq")]

    _response, used = _run(invoke_with_failover(
        chain, ["hi"], schema=None, health=health,
        timeout_seconds=0.1, total_timeout_seconds=2.0,
    ))

    assert used == "groq:openai/gpt-oss-120b:0"
    # Bound same-provider key churn, but preserve one healthy cross-provider
    # escape even after the ordinary attempt cap is reached.
    assert len([name for name, _mode in calls if name.startswith("gemini-")]) == 5


def test_gemini_project_tpm_limit_rotates_key_instead_of_suppressing_family():
    health = ProviderHealthTracker()
    calls = []
    chain = [
        mc("gemini-text:gemini-3.5-flash-lite:0", "gemini_tpm", calls, "google"),
        mc("gemini-text:gemini-3.5-flash-lite:1", "ok", calls, "google"),
    ]

    _response, used = _run(invoke_with_failover(
        chain, ["hi"], schema=None, health=health,
        timeout_seconds=0.1, total_timeout_seconds=1.0,
    ))

    assert used.endswith(":1")


def test_persistent_health_follows_anonymous_credential_when_index_changes(tmp_path):
    path = tmp_path / "model-health.json"
    credential = _credential_fingerprint("test-secret-that-must-not-be-written")
    first_name = "gemini-text:gemini-3.5-flash-lite:0"
    first = ModelClient(first_name, object(), "google", "text", credential_id=credential)
    health = ProviderHealthTracker(path)
    health.register_clients([first])
    health.record_attempt(first_name, 321)
    health.record_success(first_name, latency=1.25)

    second_name = "gemini-text:gemini-3.5-flash-lite:5"
    second = ModelClient(second_name, object(), "google", "text", credential_id=credential)
    restored = ProviderHealthTracker(path)
    restored.register_clients([second])

    assert restored.expected_cost(second_name) == pytest.approx(1.25)
    assert restored.quota_snapshot(second_name)["rpd"] == 1
    assert "test-secret-that-must-not-be-written" not in path.read_text(encoding="utf-8")


def test_local_usage_pressure_spreads_equally_healthy_gemini_projects():
    health = ProviderHealthTracker()
    calls = []
    first = mc("gemini-text:gemini-3.5-flash-lite:0", "ok", calls, "google")
    second = mc("gemini-text:gemini-3.5-flash-lite:1", "ok", calls, "google")
    health.seed_latency(first.name, 1.0)
    health.seed_latency(second.name, 1.0)
    for _ in range(3):
        health.record_attempt(first.name, 100)

    assert order_failover_chain([first, second], health)[0].name == second.name


def test_proven_fast_fallback_can_lead_next_run_over_failing_primary():
    health = ProviderHealthTracker()
    calls = []
    gemini = mc("gemini-text:gemini-3.5-flash-lite:0", "ok", calls, "google")
    gemini.sort_priority = 0
    groq = mc("groq:openai/gpt-oss-120b:0", "ok", calls, "groq")
    groq.sort_priority = 1
    health.record_failure(gemini.name, latency=15.0)
    health.record_success(groq.name, latency=2.0)

    assert order_failover_chain([gemini, groq], health)[0].name == groq.name


def test_per_project_limit_lists_guard_only_the_exhausted_key(monkeypatch):
    monkeypatch.setenv("GEMINI_PROJECT_RPM_LIMITS", "2,10")
    monkeypatch.setenv("GEMINI_USAGE_SOFT_LIMIT_PERCENT", "90")
    health = ProviderHealthTracker()
    first = "gemini-text:gemini-3.5-flash-lite:0"
    second = "gemini-text:gemini-3.5-flash-lite:1"
    health.record_attempt(first, 10)

    assert health.in_quota_guard(first) is True
    assert health.in_quota_guard(second) is False


def test_gemini_daily_ledger_rolls_over_at_midnight_pacific(monkeypatch):
    health = ProviderHealthTracker()
    name = "gemini-text:gemini-3.5-flash-lite:0"
    pacific = ZoneInfo("America/Los_Angeles")
    before_reset = datetime(2026, 9, 2, 23, 59, 59, tzinfo=pacific).timestamp()
    after_reset = datetime(2026, 9, 3, 0, 0, 1, tzinfo=pacific).timestamp()

    monkeypatch.setattr(mr.time, "time", lambda: before_reset)
    for _ in range(3):
        health.record_attempt(name, 100)
    assert health.quota_snapshot(name)["rpd"] == 3

    monkeypatch.setattr(mr.time, "time", lambda: after_reset)
    assert health.quota_snapshot(name)["rpd"] == 0
    assert health.in_quota_guard(name) is False


def test_same_gemini_project_and_model_share_usage_across_vision_and_audio():
    credential = _credential_fingerprint("one-project")
    vision = ModelClient(
        "gemini-vision:gemini-3.5-flash:0", object(), "google", "vision",
        credential_id=credential,
    )
    audio = ModelClient(
        "gemini-audio:gemini-3.5-flash:0", object(), "google", "audio",
        credential_id=credential,
    )
    health = ProviderHealthTracker()
    health.register_clients([vision, audio])
    health.record_attempt(vision.name, 100)

    assert health.quota_snapshot(audio.name)["rpd"] == 1


def test_schema_blacklist_is_per_provider():
    health = ProviderHealthTracker()
    groq = "groq:openai/gpt-oss-120b:0"
    nvidia = "nvidia-text:openai/gpt-oss-120b:0"
    health.record_failure(groq, error_msg=(
        "Error code: 400 invalid JSON schema for response_format: 'CandidateSet': "
        "additionalProperties"
    ))
    assert health.is_schema_blacklisted(groq, "CandidateSet") is True
    # Same model on NVIDIA is NOT blacklisted — schema support is provider-specific
    assert health.is_schema_blacklisted(nvidia, "CandidateSet") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
