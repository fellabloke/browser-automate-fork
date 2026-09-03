"""Deterministic run metrics for survey recovery experiments.

The benchmark is deliberately browser/provider agnostic.  Production code can
feed it events, while tests and replay fixtures can compare recovery changes
without making real provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SurveyBenchmarkMetrics:
    provider_attempts: int = 0
    model_attempts: int = 0
    vision_calls: int = 0
    vision_cache_hits: int = 0
    vision_cache_bypasses: int = 0
    duplicate_actions: int = 0
    same_state_actions: int = 0
    captcha_attempts: int = 0
    unnecessary_actions: int = 0
    valid_actions: int = 0
    final_success: bool = False
    total_elapsed_ms: float = 0.0
    first_valid_action_ms: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: dict[str, Any]) -> None:
        """Consume a normalized event without retaining raw page/provider data."""
        kind = str(event.get("kind") or event.get("classification") or "").lower()
        self.events.append({"kind": kind, "elapsed_ms": event.get("elapsed_ms")})
        if kind in {"provider_attempt", "success", "timeout", "rate_limit", "quota", "http_error", "malformed_structured_output", "transport_error"}:
            self.provider_attempts += 1
        if kind in {"model_attempt", "provider_attempt"}:
            self.model_attempts += 1
        if kind == "vision": self.vision_calls += 1
        if kind == "vision_cache_hit": self.vision_cache_hits += 1
        if kind == "vision_cache_bypass": self.vision_cache_bypasses += 1
        if kind == "duplicate_action": self.duplicate_actions += 1
        if kind == "same_state_action": self.same_state_actions += 1
        if kind == "captcha_attempt": self.captcha_attempts += 1
        if kind == "unnecessary_action": self.unnecessary_actions += 1
        if kind == "valid_action":
            self.valid_actions += 1
            if self.first_valid_action_ms is None:
                self.first_valid_action_ms = float(event.get("elapsed_ms") or 0.0)

    def summary(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "events"}
