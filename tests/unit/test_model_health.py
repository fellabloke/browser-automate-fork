"""Characterization tests for the model health/cooldown state boundary."""

from __future__ import annotations

import pytest

import agent_first_browse.models.registry as registry_module
from agent_first_browse.models.registry import (
    ModelClient,
    ProviderHealthTracker,
    order_failover_chain,
)


def test_cooldown_is_instance_scoped_and_expires(monkeypatch):
    now = 1_000.0
    monkeypatch.setattr(registry_module.time, "time", lambda: now)
    health = ProviderHealthTracker()
    first = "groq:openai/gpt-oss-120b:0"
    sibling = "groq:openai/gpt-oss-120b:1"

    health.start_cooldown(first, 30.0)

    assert health.in_cooldown(first) is True
    assert health.in_cooldown(sibling) is False
    monkeypatch.setattr(registry_module.time, "time", lambda: now + 31.0)
    assert health.in_cooldown(first) is False


def test_persisted_health_state_survives_reload_and_malformed_cache_is_ignored(tmp_path):
    path = tmp_path / "model-health.json"
    name = "groq:openai/gpt-oss-120b:0"
    health = ProviderHealthTracker(path)
    health.record_attempt(name, estimated_input_tokens=123)
    health.record_success(name, latency=1.25)

    restored = ProviderHealthTracker(path)
    assert restored.quota_snapshot(name)["rpd"] == 1
    assert restored.expected_cost(name) == pytest.approx(1.25)

    path.write_text("not-json", encoding="utf-8")
    assert ProviderHealthTracker(path)._get(name)["total_calls"] == 0


def test_probe_cache_distinguishes_fresh_and_stale_health(monkeypatch):
    now = 2_000.0
    monkeypatch.setenv("MODEL_PROBE_CACHE_SECONDS", "60")
    monkeypatch.setattr(registry_module.time, "time", lambda: now)
    health = ProviderHealthTracker()
    name = "groq:openai/gpt-oss-120b:0"

    assert health.probe_cache_fresh(name) is False
    health.record_success(name, latency=0.5)
    assert health.probe_cache_fresh(name) is True
    monkeypatch.setattr(registry_module.time, "time", lambda: now + 61.0)
    assert health.probe_cache_fresh(name) is False


def test_hard_dead_failure_class_persists_and_success_clears_it(tmp_path):
    path = tmp_path / "model-health.json"
    name = "groq:missing-model:0"
    health = ProviderHealthTracker(path)
    health.record_hard_dead(name, "MODEL_NOT_FOUND")
    restored = ProviderHealthTracker(path)
    assert restored.is_hard_dead(name)
    assert restored._get(name)["last_failure_class"] == "MODEL_NOT_FOUND"
    restored.record_success(name, latency=0.1)
    assert not restored.is_hard_dead(name)


def test_transient_failure_is_not_hard_dead():
    health = ProviderHealthTracker()
    name = "groq:model:0"
    health.record_failure(name, error_msg="429 rate limit", reliability_failure=False, failure_class="TRANSIENT")
    assert not health.is_hard_dead(name)


def test_credential_identity_preserves_health_across_key_index_changes(tmp_path):
    path = tmp_path / "model-health.json"
    credential = "stable-credential-fingerprint"
    first_name = "gemini-text:gemini-3.5-flash-lite:0"
    second_name = "gemini-text:gemini-3.5-flash-lite:5"
    first = ModelClient(first_name, object(), "google", "text", credential_id=credential)
    second = ModelClient(second_name, object(), "google", "text", credential_id=credential)

    health = ProviderHealthTracker(path)
    health.register_clients([first])
    health.record_attempt(first_name, 321)
    health.record_success(first_name, latency=1.0)

    restored = ProviderHealthTracker(path)
    restored.register_clients([second])
    assert restored.quota_snapshot(second_name)["rpd"] == 1
    assert restored.expected_cost(second_name) == pytest.approx(1.0)


def test_health_state_changes_failover_order_without_reordering_model_inputs():
    health = ProviderHealthTracker()
    first = ModelClient("groq:openai/gpt-oss-120b:0", object(), "groq", "text")
    second = ModelClient("nvidia-text:openai/gpt-oss-120b:0", object(), "nvidia", "text")
    first.sort_priority = 0
    second.sort_priority = 1
    health.record_failure(first.name, latency=15.0)
    health.record_success(second.name, latency=1.0)

    assert [client.name for client in order_failover_chain([first, second], health)] == [
        second.name,
        first.name,
    ]
