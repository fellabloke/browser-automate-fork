"""Deterministic model and role routing policy.

This module selects and orders already-constructed ``ModelClient`` instances.
It does not construct provider clients, probe models, invoke providers, or
perform retry/recovery work.
"""

from __future__ import annotations

import logging
import os

from .health import ProviderHealthTracker, base_model_name, normalize_model_id
from .schemas import ModelClient

try:
    from app.logger import get_logger

    logger = get_logger("model_registry")
except ImportError:
    logger = logging.getLogger("model_registry")


DEFAULT_MODEL_TIERS: dict[str, int] = {
    "gemini-3.5-flash-lite": 0,
    "gpt-oss-120b": 0,
    "nemotron-3.5-lightning-30b-a3b": 1,
    "llama-3.3-70b-instruct-fp8-fast": 1,
    "gemma-4-31b-it": 1,
    "gemma-4-32b-it": 1,
    "gemma-4-26b-a4b-it": 1,
    "llama-4-scout-17b-16e-instruct": 0,
    "gemini-3.5-flash": 0,
    "llama-3.2-11b-vision-instruct": 1,
    "llama-3.1-nemotron-nano-vl-8b-v1": 1,
    "llama-4-maverick-17b-128e-instruct": 1,
    "llama-3.2-90b-vision-instruct": 2,
}
UNKNOWN_MODEL_TIER = 3


def _load_tier_overrides() -> dict[str, int]:
    """Parse MODEL_TIER_OVERRIDE='model=0,other=2' from the environment."""
    overrides: dict[str, int] = {}
    raw = os.getenv("MODEL_TIER_OVERRIDE", "").strip()
    for part in raw.split(","):
        if "=" in part:
            model_id, _, rank = part.partition("=")
            try:
                overrides[normalize_model_id(model_id)] = int(rank)
            except ValueError:
                pass
    return overrides


MODEL_TIERS: dict[str, int] = {**DEFAULT_MODEL_TIERS, **_load_tier_overrides()}


def get_model_tier(instance_name: str) -> int:
    """Return the configured quality tier for a model instance name."""
    return MODEL_TIERS.get(
        normalize_model_id(base_model_name(instance_name)),
        UNKNOWN_MODEL_TIER,
    )


WORKER_MAX_TIER = 1
DEFAULT_WORKER_MODEL_ORDER = (
    "google:gemini-3.5-flash-lite,"
    "nvidia:nemotron-3.5-lightning-30b-a3b,"
    "cloudflare:llama-3.3-70b-instruct-fp8-fast,"
    "nvidia:gpt-oss-120b,"
)


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def worker_priority(model: ModelClient) -> int:
    """Resolve explicit provider/model priority for critical worker calls."""
    requested = [
        entry.strip().lower()
        for entry in os.getenv("WORKER_MODEL_ORDER", DEFAULT_WORKER_MODEL_ORDER).split(",")
        if entry.strip()
    ]
    model_name = normalize_model_id(base_model_name(model.name))
    provider = model.provider.lower()
    for priority, selector in enumerate(requested):
        selected_provider, separator, selected_model = selector.partition(":")
        if separator and selected_provider == provider and normalize_model_id(selected_model) == model_name:
            return priority
    return len(requested)


def order_failover_chain(
    entries: list[ModelClient],
    health: ProviderHealthTracker | None,
) -> list[ModelClient]:
    """Order candidates by role priority, tier, and health-informed cost."""

    def sort_key(model: ModelClient):
        tier = get_model_tier(model.name)
        use_health = bool(health and (model.credential_id or not health.has_persistence))
        cost = health.expected_cost(model.name) if use_health else 0.0
        quota = health.quota_penalty(model.name) if use_health else 0.0
        role_penalty = _float_env("MODEL_ROLE_PRIORITY_PENALTY_SECONDS", 6.0, minimum=0.0)
        tier_penalty = _float_env("MODEL_TIER_PENALTY_SECONDS", 5.0, minimum=0.0)
        score = cost + quota + model.sort_priority * role_penalty + tier * tier_penalty
        return (score, tier, model.sort_priority)

    return sorted(entries, key=sort_key)


def route_worker_chain(
    pipeline: list[ModelClient],
    health: ProviderHealthTracker,
    mode: str,
) -> list[ModelClient]:
    """Select and order the worker candidates without invoking any provider."""
    if mode == "premium":
        return list(pipeline)
    top = [model for model in pipeline if get_model_tier(model.name) <= WORKER_MAX_TIER]
    selected = top or list(pipeline)
    reliable = [
        model
        for model in selected
        if not (
            (model.credential_id or not health.has_persistence)
            and health.is_chronically_unreliable(model.name)
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
            name=model.name,
            client=model.client,
            provider=model.provider,
            pipeline=model.pipeline,
            sort_priority=worker_priority(model),
            credential_id=model.credential_id,
            critical=True,
        )
        for model in selected
    ]
    return order_failover_chain(prioritized, health)


def route_auxiliary_chain(
    pipeline: list[ModelClient],
    mode: str,
) -> list[ModelClient]:
    """Shape auxiliary candidates by configured provider priority."""
    if mode == "premium":
        return list(pipeline)
    requested = [
        provider.strip().lower()
        for provider in os.getenv(
            "AUXILIARY_PROVIDER_ORDER", "google,cloudflare,nvidia"
        ).split(",")
        if provider.strip()
    ]
    rank = {provider: index for index, provider in enumerate(requested)}
    fallback_rank = len(rank)
    return [
        ModelClient(
            name=model.name,
            client=model.client,
            provider=model.provider,
            pipeline=model.pipeline,
            sort_priority=rank.get(model.provider.lower(), fallback_rank),
            credential_id=model.credential_id,
        )
        for model in pipeline
    ]


def route_auxiliary_chain_names(
    pipeline: list[ModelClient],
    health: ProviderHealthTracker,
    mode: str,
) -> list[str]:
    """Return auxiliary candidates in their deterministic health-aware order."""
    return [
        model.name
        for model in order_failover_chain(route_auxiliary_chain(pipeline, mode), health)
    ]
