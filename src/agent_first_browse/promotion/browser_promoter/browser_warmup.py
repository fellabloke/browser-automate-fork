"""Compatibility exports for the canonical browser warm-up implementation."""

from agent_first_browse.browser.warmup import (
    extract_target_url_from_objective,
    run_warmup,
)

__all__ = ["extract_target_url_from_objective", "run_warmup"]
