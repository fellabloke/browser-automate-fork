"""Shared model-layer data contracts."""

from __future__ import annotations

from typing import Any


class ModelClient:
    """Named provider client with routing and credential metadata."""

    __slots__ = (
        "name", "client", "provider", "pipeline", "sort_priority", "credential_id",
        "critical",
    )

    def __init__(
        self,
        name: str,
        client: Any,
        provider: str,
        pipeline: str,
        sort_priority: int = 0,
        credential_id: str = "",
        critical: bool = False,
    ):
        self.name = name
        self.client = client
        self.provider = provider
        self.pipeline = pipeline
        self.sort_priority = sort_priority
        self.credential_id = credential_id
        self.critical = critical

    def __repr__(self) -> str:
        return f"ModelClient({self.name}, pipeline={self.pipeline})"
