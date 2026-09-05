"""Characterization tests for model/provider construction."""

from __future__ import annotations

import sys
import types

import pytest

import agent_first_browse.models.registry as mr


class FakeChatClient:
    created: list[dict] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.openai_api_base = kwargs.get("base_url", "")
        self.max_tokens = kwargs.get("max_tokens")
        self.__class__.created.append(kwargs)


class FakeGoogleClient:
    created: list[dict] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.__class__.created.append(kwargs)


@pytest.fixture
def fake_provider_sdks(monkeypatch):
    FakeChatClient.created = []
    FakeGoogleClient.created = []
    openai = types.ModuleType("langchain_openai")
    openai.ChatOpenAI = FakeChatClient
    google = types.ModuleType("langchain_google_genai")
    google.ChatGoogleGenerativeAI = FakeGoogleClient
    monkeypatch.setitem(sys.modules, "langchain_openai", openai)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", google)
    return FakeChatClient, FakeGoogleClient


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    names = (
        "NVIDIA_NIM_API_KEY", "NVIDIA_NIM_API_KEYS", "NVIDIA_NIM_BASE_URL",
        "NVIDIA_TEXT_MODELS", "NVIDIA_VISION_API_KEY", "NVIDIA_VISION_MODELS",
        "NVIDIA_VISION_BASE_URL", "NVIDIA_VISION_TIMEOUT",
        "GEMINI_API_KEY", "GEMINI_API_KEY_FALLBACKS", "GEMINI_TEXT_MODEL",
        "WORKER_VLM_MODEL", "VISION_GOOGLE_API_KEY", "VISION_GOOGLE_API_KEY_FALLBACKS",
        "GOOGLE_API_KEY", "GOOGLE_API_KEY_FALLBACKS", "SURVEY_AUDIO_ENABLED",
        "SURVEY_AUDIO_MODEL", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_API_TOKENS", "CLOUDFLARE_ENABLED", "CLOUDFLARE_VISION_ENABLED",
        "CLOUDFLARE_BASE_URL", "CLOUDFLARE_TEXT_MODELS", "CLOUDFLARE_VISION_MODELS",
        "TEXT_MODEL_MAX_TOKENS", "CLOUDFLARE_MAX_TOKENS",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_text_construction_preserves_provider_order_and_metadata(monkeypatch, fake_provider_sdks):
    monkeypatch.setenv("NVIDIA_NIM_API_KEYS", "nv-a,nv-b")
    monkeypatch.setenv("NVIDIA_TEXT_MODELS", "nvidia/model-a,nvidia/model-b")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-1")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-a")

    clients = mr._build_text_pipeline()

    assert [client.provider for client in clients] == [
        "nvidia", "nvidia", "nvidia", "nvidia", "cloudflare", "google",
    ]
    assert [client.pipeline for client in clients] == ["text"] * len(clients)
    assert [client.name for client in clients[:4]] == [
        "nvidia-text:nvidia/model-a:0",
        "nvidia-text:nvidia/model-b:0",
        "nvidia-text:nvidia/model-a:1",
        "nvidia-text:nvidia/model-b:1",
    ]
    assert clients[0].critical is False
    assert clients[0].sort_priority == 0
    assert clients[0].credential_id != "nv-a"
    assert clients[-1].name == "gemini-text:gemini-3.5-flash-lite:0"


def test_vision_and_audio_construction_preserve_pipeline_membership(monkeypatch, fake_provider_sdks):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-a")
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nv-a")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-1")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("SURVEY_AUDIO_MODEL", "gemini-audio-test")

    vision = mr._build_vision_pipeline()
    audio = mr._build_audio_pipeline()

    assert [client.provider for client in vision] == [
        "google", "nvidia", "nvidia", "nvidia", "nvidia", "cloudflare",
    ]
    assert all(client.pipeline == "vision" for client in vision)
    assert [client.name for client in audio] == ["gemini-audio:gemini-audio-test:0"]
    assert audio[0].pipeline == "audio"


def test_missing_credentials_and_optional_google_dependency_are_tolerated(monkeypatch, fake_provider_sdks):
    assert mr._build_text_pipeline() == []
    assert mr._build_vision_pipeline() == []
    assert mr._build_audio_pipeline() == []

    monkeypatch.setenv("GEMINI_API_KEY", "gem-a")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    assert mr._build_text_pipeline() == []
    assert mr._build_vision_pipeline() == []


def test_duplicate_models_are_retained_in_current_configuration_behavior(
    monkeypatch, fake_provider_sdks
):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nv-a")
    monkeypatch.setenv("NVIDIA_TEXT_MODELS", "nvidia/model-a,nvidia/model-a")

    clients = mr._build_text_pipeline()

    assert [client.name for client in clients] == [
        "nvidia-text:nvidia/model-a:0",
        "nvidia-text:nvidia/model-a:0",
    ]


def test_premium_construction_preserves_key_instances_and_provider_metadata(
    monkeypatch, fake_provider_sdks
):
    monkeypatch.setenv("PREMIUM_API_KEYS", "premium-a,premium-b")
    monkeypatch.setenv("PREMIUM_MODEL", "premium-model")
    monkeypatch.setenv("PREMIUM_BASE_URL", "https://gateway.test/v1")
    monkeypatch.setenv("PREMIUM_PROVIDER", "openai")

    text, vision = mr._build_premium_pipeline()

    assert [client.name for client in text] == [
        "premium-text:premium-model:0",
        "premium-text:premium-model:1",
    ]
    assert [client.name for client in vision] == [
        "premium-vision:premium-model:0",
        "premium-vision:premium-model:1",
    ]
    assert all(client.provider == "premium" for client in text + vision)
    assert all(client.credential_id for client in text + vision)


def test_cloudflare_vision_builder_returns_native_adapter_without_network(
    monkeypatch, fake_provider_sdks
):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account-1")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_VISION_MODELS", "@cf/model-a,@cf/model-b")

    clients = mr._build_vision_pipeline()
    cloudflare = [client for client in clients if client.provider == "cloudflare"]

    assert [client.name for client in cloudflare] == [
        "cloudflare-vision:@cf/model-a:0",
        "cloudflare-vision:@cf/model-b:0",
    ]
    assert all(isinstance(client.client, mr.CloudflareNativeVisionClient) for client in cloudflare)
    assert all(client.client.account_id == "account-1" for client in cloudflare)
