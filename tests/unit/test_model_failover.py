"""Characterization tests for ordinary model invocation and recovery."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from agent_first_browse.models.health import ProviderHealthTracker
from agent_first_browse.models.registry import ModelClient, invoke_with_failover


class _Response:
    def __init__(self, content: str):
        self.content = content


class _Tiny(BaseModel):
    ok: bool


class _RecordingClient:
    def __init__(self, name: str, *, strict: str = "ok", raw: str = "ok"):
        self.name = name
        self.strict = strict
        self.raw = raw
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        parent = self

        class _Structured:
            async def ainvoke(self, messages):
                parent.calls.append("strict")
                if parent.strict == "malformed":
                    raise ValueError("validation error: invalid structured output")
                if parent.strict != "ok":
                    raise RuntimeError(parent.strict)
                return _Response('{"ok":true}')

        return _Structured()

    async def ainvoke(self, messages):
        self.calls.append("raw")
        if self.raw != "ok":
            raise RuntimeError(self.raw)
        return _Response('{"ok":true}')


def _client(name: str, fake: _RecordingClient, provider: str = "groq") -> ModelClient:
    return ModelClient(name=name, client=fake, provider=provider, pipeline="text")


def run(coro):
    return asyncio.run(coro)


def test_successful_first_attempt_records_health_and_affinity():
    fake = _RecordingClient("primary")
    client = _client("groq:openai/gpt-oss-120b:0", fake)
    health = ProviderHealthTracker()

    response, used = run(invoke_with_failover([client], ["hello"], health=health))

    assert response.content == '{"ok":true}'
    assert used == client.name
    assert fake.calls == ["raw"]
    assert health.preferred_for_role("TEXT_WORKER", {client.name}) == client.name
    assert health.quota_snapshot(client.name)["rpd"] == 1


def test_failover_consumes_supplied_sibling_then_provider_order():
    first = _RecordingClient("first", raw="transport failure")
    sibling = _RecordingClient("sibling")
    fallback = _RecordingClient("fallback")
    chain = [
        _client("groq:openai/gpt-oss-120b:0", first, "groq"),
        _client("groq:openai/gpt-oss-120b:1", sibling, "groq"),
        _client("nvidia-text:openai/gpt-oss-120b:0", fallback, "nvidia"),
    ]

    _, used = run(invoke_with_failover(chain, ["hello"], health=ProviderHealthTracker()))

    assert used == chain[1].name
    assert first.calls == ["raw"]
    assert sibling.calls == ["raw"]
    assert fallback.calls == []


def test_malformed_structured_output_repairs_on_same_client_before_failover():
    repairing = _RecordingClient("repairing", strict="malformed", raw="ok")
    sibling = _RecordingClient("sibling")
    first = _client("groq:openai/gpt-oss-120b:0", repairing)
    second = _client("groq:openai/gpt-oss-120b:1", sibling)
    health = ProviderHealthTracker()

    response, used = run(invoke_with_failover([first, second], ["hello"], _Tiny, health=health))

    assert response.ok is True
    assert used == first.name
    assert repairing.calls == ["strict", "raw"]
    assert sibling.calls == []
    assert health.is_schema_blacklisted(first.name, "_Tiny")
    assert health._get(first.name)["structured_repair_successes"] == 1


def test_failed_same_model_repair_precedes_next_candidate():
    repairing = _RecordingClient("repairing", strict="malformed", raw="repair failure")
    sibling = _RecordingClient("sibling")
    first = _client("groq:openai/gpt-oss-120b:0", repairing)
    second = _client("groq:openai/gpt-oss-120b:1", sibling)
    health = ProviderHealthTracker()

    _, used = run(invoke_with_failover([first, second], ["hello"], _Tiny, health=health))

    assert used == second.name
    assert repairing.calls == ["strict", "raw"]
    # The provider/model schema blacklist is family-scoped, so the sibling
    # starts directly in JSON mode after the first client's failed repair.
    assert sibling.calls == ["raw"]
    assert health._get(first.name)["structured_repair_failures"] == 1


def test_forced_json_mode_skips_strict_invocation():
    fake = _RecordingClient("forced")
    client = _client("groq:openai/gpt-oss-120b:0", fake)
    health = ProviderHealthTracker()
    health.force_json_mode(client.name, "_Tiny")

    response, used = run(invoke_with_failover([client], ["hello"], _Tiny, health=health))

    assert response.ok is True
    assert used == client.name
    assert fake.calls == ["raw"]


def test_terminal_failure_preserves_sanitized_provider_diagnostic():
    fake = _RecordingClient("secret", raw="429 rate limit for gsk_super-secret")
    client = _client("groq:openai/gpt-oss-120b:0", fake)

    with pytest.raises(RuntimeError) as error:
        run(invoke_with_failover([client], ["hello"], total_timeout_seconds=1.0))

    assert "gsk_super-secret" not in str(error.value)
    assert "<redacted-key>" in str(error.value)
