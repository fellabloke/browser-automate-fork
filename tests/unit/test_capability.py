"""Unit tests for Agentic Capability Gate + dual-mode (free/premium).

Guarantees under test (mirror ARCHITECTURE.md invariants):
  - Mode detection: auto→premium iff PREMIUM_API_KEY; explicit premium/free force.
  - Premium pipeline: one key/model serves text+vision; separate vision model
    honored; missing model → empty (caller falls back to free).
  - Registry in premium mode skips the probe (trusted) and the worker chain is
    the single premium model.
  - Premium misconfigured → safe fallback to free.
  - Role separation: worker chain is tier ≤ WORKER_MAX_TIER, never empty.
  - Capability gate: incapable combos excluded; if that would EMPTY a pipeline,
    allowlisted-alive models are restored (never empty — invariant #2).

Run: .venv/bin/python -m pytest tests/unit/test_capability.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]

from agent_first_browse.models.registry import (
    AGENTIC_TEXT_ALLOWLIST,
    WORKER_MAX_TIER,
    CloudflareNativeVisionClient,
    ModelClient,
    ModelRegistry,
    _build_premium_pipeline,
    _build_text_pipeline,
    get_agent_mode,
    get_model_tier,
    order_failover_chain,
)


def _mc(name: str, provider: str, pipeline: str = "text") -> ModelClient:
    return ModelClient(name=name, client=None, provider=provider, pipeline=pipeline)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PREMIUM_API_KEY", "PREMIUM_API_KEYS", "PREMIUM_MODEL",
              "PREMIUM_VISION_MODEL", "PREMIUM_BASE_URL", "PREMIUM_PROVIDER", "AGENT_MODE",
              "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN",
              "CLOUDFLARE_API_TOKENS", "CLOUDFLARE_BASE_URL",
              "CLOUDFLARE_TEXT_MODELS", "CLOUDFLARE_ENABLED",
              "CLOUDFLARE_MAX_TOKENS", "AUXILIARY_PROVIDER_ORDER",
              "WORKER_MODEL_ORDER"):
        monkeypatch.delenv(k, raising=False)
    yield
    ModelRegistry.reset()


# ═══════════════════════════════════════════════════════════════════════════════
#  Mode detection
# ═══════════════════════════════════════════════════════════════════════════════

def test_auto_is_free_without_premium_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "auto")
    assert get_agent_mode() == "free"


def test_auto_is_premium_with_premium_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "auto")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    assert get_agent_mode() == "premium"


def test_explicit_free_overrides_premium_key(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    assert get_agent_mode() == "free"


def test_explicit_premium_forces(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "premium")
    assert get_agent_mode() == "premium"


# ═══════════════════════════════════════════════════════════════════════════════
#  Premium pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def test_premium_one_key_serves_text_and_vision(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    monkeypatch.setenv("PREMIUM_MODEL", "gpt-5")
    text, vision = _build_premium_pipeline()
    assert len(text) == 1 and len(vision) == 1
    assert text[0].pipeline == "text" and vision[0].pipeline == "vision"
    assert text[0].provider == "premium"
    assert "gpt-5" in text[0].name and "gpt-5" in vision[0].name


def test_premium_separate_vision_model(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    monkeypatch.setenv("PREMIUM_MODEL", "claude-opus-4")
    monkeypatch.setenv("PREMIUM_VISION_MODEL", "claude-vision")
    text, vision = _build_premium_pipeline()
    assert "claude-opus-4" in text[0].name
    assert "claude-vision" in vision[0].name


def test_premium_missing_model_returns_empty(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")  # no PREMIUM_MODEL
    text, vision = _build_premium_pipeline()
    assert text == [] and vision == []


def test_premium_multiple_keys(monkeypatch):
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-a,sk-b")
    monkeypatch.setenv("PREMIUM_MODEL", "gpt-5")
    text, _ = _build_premium_pipeline()
    assert len(text) == 2  # one instance per key for headroom


def test_registry_premium_skips_probe(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "premium")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")
    monkeypatch.setenv("PREMIUM_MODEL", "gpt-5")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    assert r.mode == "premium"
    assert r._probed is True                       # trusted — never probes
    assert r.get_worker_chain_names() == ["premium-text:gpt-5:0"]
    assert r.get_vision_chain_names() == ["premium-vision:gpt-5:0"]


def test_registry_premium_misconfigured_falls_back_to_free(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "premium")
    monkeypatch.setenv("PREMIUM_API_KEY", "sk-x")   # but NO PREMIUM_MODEL
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    assert r.mode == "free"                          # safe fallback
    assert r._probed is False                        # free mode will probe


# ═══════════════════════════════════════════════════════════════════════════════
#  Role separation (worker chain)
# ═══════════════════════════════════════════════════════════════════════════════

def test_worker_chain_only_top_tier(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    r._text_pipeline = [
        _mc("nvidia-text:openai/gpt-oss-120b:0", "nvidia"),  # tier 0
        _mc("nvidia-text:openai/gpt-oss-20b:0", "nvidia"),  # retired — excluded
        _mc("legacy:retired-model:0", "legacy"),          # tier 2 — excluded from worker
        _mc("x:glm-5.1:0", "x"),                         # tier 2 — excluded
    ]
    wc = r.get_worker_chain_names()
    assert all(get_model_tier(n) <= WORKER_MAX_TIER for n in wc), wc
    assert "nvidia-text:openai/gpt-oss-120b:0" in wc
    assert "legacy:retired-model:0" not in wc


def test_worker_chain_never_empty(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    r._text_pipeline = [_mc("legacy:retired-model:0", "legacy")]  # only tier 2
    # No tier-0/1 model → worker must fall back to the full chain, not empty.
    assert r.get_worker_chain_names() == ["legacy:retired-model:0"]


def test_worker_chain_uses_explicit_optimal_model_order(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    r._text_pipeline = [
        _mc("nvidia-text:openai/gpt-oss-120b:0", "nvidia"),
        _mc("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast:0", "cloudflare"),
        _mc("nvidia-text:nvidia/nemotron-3.5-lightning-30b-a3b:0", "nvidia"),
        _mc("gemini-text:gemini-3.5-flash-lite:0", "google"),
    ]

    assert r.get_worker_chain_names() == [
        "gemini-text:gemini-3.5-flash-lite:0",
        "nvidia-text:nvidia/nemotron-3.5-lightning-30b-a3b:0",
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast:0",
        "nvidia-text:openai/gpt-oss-120b:0",
    ]


def test_cloudflare_text_pipeline_uses_account_endpoint(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
    monkeypatch.setenv("CLOUDFLARE_BASE_URL", "")
    monkeypatch.setenv(
        "CLOUDFLARE_TEXT_MODELS",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    )
    monkeypatch.setenv("CLOUDFLARE_MAX_TOKENS", "2048")
    monkeypatch.setenv("TEXT_MODEL_MAX_TOKENS", "1000")

    cloudflare = [m for m in _build_text_pipeline() if m.provider == "cloudflare"]

    assert [m.name for m in cloudflare] == [
        "cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast:0",
    ]
    assert all(m.pipeline == "text" for m in cloudflare)
    assert all("account-123/ai/v1" in str(m.client.openai_api_base) for m in cloudflare)
    # The global structured-action ceiling wins over a provider-specific
    # larger allowance so free-tier TPM is not wasted on unused completion.
    assert all(m.client.max_tokens == 1000 for m in cloudflare)


def test_cloudflare_token_without_account_is_ignored(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
    assert not [m for m in _build_text_pipeline() if m.provider == "cloudflare"]


@pytest.mark.asyncio
async def test_cloudflare_native_vision_preserves_multimodal_messages(monkeypatch):
    import httpx
    from langchain_core.messages import HumanMessage

    captured = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "success": True,
                "result": {
                    "response": '{"color":"blue"}',
                    "usage": {"neurons": 1.25},
                },
            }

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, headers, json):
            captured.update(url=url, headers=headers, payload=json)
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    class ColorResult(BaseModel):
        color: str

    client = CloudflareNativeVisionClient(
        account_id="account-123",
        api_token="cfut_test-token",
        model="@cf/meta/llama-3.2-11b-vision-instruct",
    ).with_structured_output(ColorResult)
    message = HumanMessage(content=[
        {"type": "text", "text": "Name the color."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,abc123"},
        },
    ])

    result = await client.ainvoke([message])

    assert result.color == "blue"
    assert captured["url"].endswith(
        "/ai/run/@cf/meta/llama-3.2-11b-vision-instruct"
    )
    user_message = captured["payload"]["messages"][-1]
    assert user_message["role"] == "user"
    assert user_message["content"][1]["image_url"]["url"].endswith("abc123")
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert captured["headers"]["Authorization"] == "Bearer cfut_test-token"


@pytest.mark.asyncio
async def test_cloudflare_native_vision_repairs_ignored_json_mode(monkeypatch):
    import httpx
    from langchain_core.messages import HumanMessage

    requests = []

    class FakeResponse:
        status_code = 200

        def __init__(self, response):
            self._response = response

        def json(self):
            return {"success": True, "result": {"response": self._response}}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, headers, json):
            requests.append(json)
            if len(requests) == 1:
                return FakeResponse("The screenshot shows a blue button.")
            return FakeResponse({"observation": "The screenshot shows a blue button."})

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    class Observation(BaseModel):
        observation: str

    client = CloudflareNativeVisionClient(
        account_id="account-123",
        api_token="cfut_test-token",
        model="@cf/meta/llama-3.2-11b-vision-instruct",
    ).with_structured_output(Observation)

    result = await client.ainvoke([HumanMessage(content="Describe the screenshot.")])

    assert result.observation == "The screenshot shows a blue button."
    assert len(requests) == 2
    assert "image_url" not in str(requests[1])


def test_auxiliary_chain_prefers_configured_provider_order(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "free")
    monkeypatch.setenv("AUXILIARY_PROVIDER_ORDER", "google,cloudflare,groq")
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    r._text_pipeline = [
        _mc("groq:openai/gpt-oss-120b:0", "groq"),
        _mc("gemini-text:gemini-3.5-flash-lite:0", "google"),
        _mc("cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast:0", "cloudflare"),
    ]

    ordered = order_failover_chain(r.get_auxiliary_chain(), r.health)
    assert [m.provider for m in ordered] == ["google", "cloudflare", "groq"]
    assert r.get_auxiliary_chain_names() == [m.name for m in ordered]


# ═══════════════════════════════════════════════════════════════════════════════
#  Capability gate
# ═══════════════════════════════════════════════════════════════════════════════

def test_gate_excludes_incapable(monkeypatch):
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    pipe = [
        _mc("groq:openai/gpt-oss-120b:0", "groq"),
        _mc("nvidia-text:google/gemma-4-31b-it:0", "nvidia"),
    ]
    incapable = {("nvidia", "gemma-4-31b-it", "text")}   # gemma-on-NVIDIA can't structure
    out = [m.name for m in r._apply_capability_gate(pipe, set(), incapable, "TEXT")]
    assert "groq:openai/gpt-oss-120b:0" in out
    assert "nvidia-text:google/gemma-4-31b-it:0" not in out


def test_gate_excludes_dead(monkeypatch):
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    pipe = [
        _mc("nvidia-text:openai/gpt-oss-120b:0", "nvidia"),
        _mc("nvidia-vision:google/gemma-4-32b-it:0", "nvidia", "vision"),
    ]
    dead = {("nvidia", "gemma-4-32b-it", "vision")}
    out = [m.name for m in r._apply_capability_gate(pipe, dead, set(), "VISION")]
    assert "nvidia-vision:google/gemma-4-32b-it:0" not in out
    assert "nvidia-text:openai/gpt-oss-120b:0" in out


def test_gate_safety_floor_never_empties(monkeypatch):
    # If the gate would remove EVERY model, restore allowlisted-alive ones.
    ModelRegistry.reset()
    r = ModelRegistry.get_instance()
    pipe = [_mc("groq:openai/gpt-oss-120b:0", "groq")]  # allowlisted
    incapable = {("groq", "gpt-oss-120b", "text")}      # (falsely) flagged incapable
    out = r._apply_capability_gate(pipe, set(), incapable, "TEXT")
    assert len(out) == 1, "pipeline must never be emptied by the gate"


def test_allowlist_has_proven_models():
    for m in (
        "gemini-3.5-flash-lite",
        "gpt-oss-120b",
        "nemotron-3.5-lightning-30b-a3b",
        "llama-3.3-70b-instruct-fp8-fast",
    ):
        assert m in AGENTIC_TEXT_ALLOWLIST


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
