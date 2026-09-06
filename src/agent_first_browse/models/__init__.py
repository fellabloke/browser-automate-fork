"""Stable public API for model construction, routing, and invocation."""

from .failover import CircuitBreaker, invoke_with_failover
from .health import ProviderHealthTracker, normalize_model_id
from .providers import CloudflareNativeVisionClient
from .registry import AGENTIC_TEXT_ALLOWLIST, ModelRegistry, get_agent_mode
from .routing import get_model_tier, order_failover_chain
from .schemas import ModelClient

__all__ = [
    "AGENTIC_TEXT_ALLOWLIST",
    "CircuitBreaker",
    "CloudflareNativeVisionClient",
    "ModelClient",
    "ModelRegistry",
    "ProviderHealthTracker",
    "get_agent_mode",
    "get_model_tier",
    "invoke_with_failover",
    "normalize_model_id",
    "order_failover_chain",
]
