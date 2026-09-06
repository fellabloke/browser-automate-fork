"""Characterization tests for deterministic startup model probing."""

from __future__ import annotations

import asyncio

from agent_first_browse.models.health import ProviderHealthTracker
from agent_first_browse.models.registry import ModelClient, ModelRegistry


class _Response:
    def __init__(self, content: str = "ok"):
        self.content = content


class _ProbeClient:
    def __init__(self, *, strict: str = "ok", raw: str = '{"action":"click"}'):
        self.strict = strict
        self.raw = raw
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        parent = self

        class _Structured:
            async def ainvoke(self, messages):
                parent.calls.append("strict")
                if parent.strict != "ok":
                    raise RuntimeError(parent.strict)
                return _Response()

        return _Structured()

    async def ainvoke(self, messages):
        self.calls.append("raw")
        if self.raw != "ok":
            raise RuntimeError(self.raw)
        return _Response('{"action":"click","element_id":"e1","confidence":0.9}')


def _registry(clients: list[ModelClient], *, health=None) -> ModelRegistry:
    registry = object.__new__(ModelRegistry)
    registry._text_pipeline = clients
    registry._vision_pipeline = []
    registry._probed = False
    registry._vision_probed = False
    registry.health = health or ProviderHealthTracker()
    return registry


def _client(name: str, fake: _ProbeClient, provider: str = "groq") -> ModelClient:
    return ModelClient(name=name, client=fake, provider=provider, pipeline="text")


def run(coro):
    return asyncio.run(coro)


def test_probe_uses_one_stable_representative_and_seeds_siblings():
    first = _ProbeClient()
    sibling = _ProbeClient()
    first_client = _client("groq:openai/gpt-oss-120b:0", first)
    sibling_client = _client("groq:openai/gpt-oss-120b:1", sibling)
    registry = _registry([first_client, sibling_client])

    run(registry.probe_and_prune(timeout=0.2))

    assert first.calls == ["strict"]
    assert sibling.calls == []
    assert registry._text_pipeline == [first_client, sibling_client]
    assert registry.health.probe_cache_fresh(first_client.name)
    assert registry.health.probe_cache_fresh(sibling_client.name)


def test_fresh_probe_cache_skips_provider_calls():
    fake = _ProbeClient()
    client = _client("groq:openai/gpt-oss-120b:0", fake)
    health = ProviderHealthTracker()
    health.record_success(client.name, latency=0.1)
    registry = _registry([client], health=health)

    run(registry.probe_and_prune(timeout=0.2))

    assert fake.calls == []
    assert registry._text_pipeline == [client]


def test_strict_schema_rejection_uses_json_rescue_and_records_requirement():
    fake = _ProbeClient(strict="strict schema unsupported", raw="ok")
    client = _client("nvidia-text:openai/gpt-oss-120b:0", fake, "nvidia")
    registry = _registry([client])

    run(registry.probe_and_prune(timeout=0.2))

    assert fake.calls == ["strict", "raw"]
    assert registry.health.is_schema_blacklisted(client.name, "__structured_output__")
    assert registry._text_pipeline == [client]


def test_dead_combo_is_pruned_but_transient_and_timeout_are_retained():
    dead = _client(
        "dead:openai/gpt-oss-120b:0", _ProbeClient(strict="404 model not found"), "dead"
    )
    transient = _client(
        "rate:openai/gpt-oss-120b:0", _ProbeClient(strict="429 rate limit"), "rate"
    )
    slow = _client(
        "slow:openai/gpt-oss-120b:0", _ProbeClient(strict="unused"), "slow"
    )

    class _Slow(_ProbeClient):
        def with_structured_output(self, schema):
            parent = self

            class _Structured:
                async def ainvoke(self, messages):
                    parent.calls.append("strict")
                    await asyncio.sleep(0.1)

            return _Structured()

    slow.client = _Slow()
    registry = _registry([dead, transient, slow])

    run(registry.probe_and_prune(timeout=0.01))

    assert [client.name for client in registry._text_pipeline] == [
        transient.name,
        slow.name,
    ]
    assert registry.health.probe_cache_fresh(transient.name) is False


def test_startup_probe_does_not_probe_vision_until_demand():
    fake = _ProbeClient()
    vision = ModelClient("vision:model:0", fake, "vision-provider", "vision")
    registry = _registry([], health=ProviderHealthTracker())
    registry._vision_pipeline = [vision]
    run(registry.probe_and_prune(timeout=0.2, probe_vision=False))
    assert fake.calls == []


def test_lazy_vision_probe_is_cached_after_first_demand():
    fake = _ProbeClient()
    vision = ModelClient("vision:model:0", fake, "vision-provider", "vision")
    registry = _registry([], health=ProviderHealthTracker())
    registry._vision_pipeline = [vision]
    run(registry.ensure_vision_capability(timeout=0.2))
    run(registry.ensure_vision_capability(timeout=0.2))
    assert fake.calls == ["strict"]
