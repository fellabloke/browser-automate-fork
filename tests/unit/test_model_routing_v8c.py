"""Characterization tests for deterministic model/role routing."""

from __future__ import annotations

import agent_first_browse.models.registry as mr
from agent_first_browse.models.health import ProviderHealthTracker
from agent_first_browse.models.schemas import ModelClient


def _client(name: str, provider: str, *, priority: int = 0) -> ModelClient:
    class NoInferenceClient:
        def __getattr__(self, name):
            raise AssertionError(f"routing attempted provider operation: {name}")

    return ModelClient(
        name=name,
        client=NoInferenceClient(),
        provider=provider,
        pipeline="text",
        sort_priority=priority,
    )


def _registry(pipeline: list[ModelClient], *, mode: str = "free") -> mr.ModelRegistry:
    registry = object.__new__(mr.ModelRegistry)
    registry.mode = mode
    registry._text_pipeline = pipeline
    registry.health = ProviderHealthTracker()
    return registry


def test_model_tiers_and_unknown_fallback_are_deterministic():
    assert mr.get_model_tier("nvidia-text:openai/gpt-oss-120b:0") == 0
    assert mr.get_model_tier("nvidia-text:google/gemma-4-31b-it:0") == 1
    assert mr.get_model_tier("x:unknown-model:0") == mr.UNKNOWN_MODEL_TIER


def test_worker_routing_selects_eligible_tiers_and_preserves_sibling_candidates():
    pipeline = [
        _client("legacy:retired-model:0", "legacy"),
        _client("nvidia-text:openai/gpt-oss-120b:0", "nvidia"),
        _client("nvidia-text:openai/gpt-oss-120b:1", "nvidia"),
        _client("groq:openai/gpt-oss-120b:0", "groq"),
    ]

    routed = _registry(pipeline).get_worker_chain()

    assert [client.name for client in routed] == [
        "nvidia-text:openai/gpt-oss-120b:0",
        "nvidia-text:openai/gpt-oss-120b:1",
        "groq:openai/gpt-oss-120b:0",
    ]
    assert all(client.critical for client in routed)
    assert all(client.pipeline == "text" for client in routed)


def test_worker_routing_falls_back_to_full_pipeline_when_no_preferred_tier_exists():
    pipeline = [_client("legacy:retired-model:0", "legacy")]

    assert [client.name for client in _registry(pipeline).get_worker_chain()] == [
        "legacy:retired-model:0"
    ]


def test_premium_worker_routing_does_not_filter_or_reorder_candidates():
    pipeline = [
        _client("premium-text:premium-model:0", "premium", priority=4),
        _client("premium-text:premium-model:1", "premium", priority=0),
    ]

    routed = _registry(pipeline, mode="premium").get_worker_chain()

    assert routed == pipeline


def test_auxiliary_routing_applies_provider_order_without_inference(monkeypatch):
    monkeypatch.setenv("AUXILIARY_PROVIDER_ORDER", "google,cloudflare,groq")
    pipeline = [
        _client("groq:openai/gpt-oss-120b:0", "groq"),
        _client("gemini-text:gemini-3.5-flash-lite:0", "google"),
        _client("cloudflare:@cf/model:0", "cloudflare"),
    ]
    registry = _registry(pipeline)

    auxiliary = registry.get_auxiliary_chain()

    assert [client.provider for client in auxiliary] == ["groq", "google", "cloudflare"]
    assert registry.get_auxiliary_chain_names() == [
        "gemini-text:gemini-3.5-flash-lite:0",
        "groq:openai/gpt-oss-120b:0",
        "cloudflare:@cf/model:0",
    ]
    assert [client.sort_priority for client in auxiliary] == [2, 0, 1]


def test_auxiliary_names_are_health_ordered_but_candidates_remain_distinct(monkeypatch):
    monkeypatch.setenv("AUXILIARY_PROVIDER_ORDER", "google,groq")
    first = _client("gemini-text:gemini-3.5-flash-lite:0", "google")
    sibling = _client("gemini-text:gemini-3.5-flash-lite:1", "google")
    fallback = _client("groq:openai/gpt-oss-120b:0", "groq")
    registry = _registry([fallback, first, sibling])
    registry.health.record_failure(first.name, latency=15.0)
    registry.health.record_success(sibling.name, latency=1.0)

    names = registry.get_auxiliary_chain_names()

    assert set(names) == {first.name, sibling.name, fallback.name}
    assert names[0] == sibling.name
    assert first.name in names and fallback.name in names


def test_routing_does_not_consume_role_affinity_or_make_provider_calls():
    pipeline = [
        _client("groq:openai/gpt-oss-120b:0", "groq"),
        _client("nvidia-text:openai/gpt-oss-120b:0", "nvidia"),
    ]
    registry = _registry(pipeline)
    baseline = [client.name for client in registry.get_worker_chain()]
    registry.health.set_preferred_for_role("TEXT_WORKER", pipeline[0].name)

    routed = registry.get_worker_chain()

    assert [client.name for client in routed] == baseline
    assert registry.health.preferred_for_role("TEXT_WORKER", {pipeline[0].name}) == pipeline[0].name
