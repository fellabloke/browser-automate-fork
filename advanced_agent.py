"""Compatibility façade for the retired autonomous-agent implementation."""

from __future__ import annotations

from agent_first_browse.browser.runtime import (
    PROFILE_DIR,
    PERSISTENCE_ROOT,
    SessionGuard,
    launch_browser,
    manual_login_mode,
    shutdown_browser,
)
from agent_first_browse.models import invoke_with_failover as _invoke_model


async def _invoke_with_failover(chain, messages, schema=None, breaker=None, **kwargs):
    """Preserve the legacy positional-breaker callback contract."""
    return await _invoke_model(chain, messages, schema, breaker=breaker, **kwargs)


async def run_agent(objective: str):
    """Run the canonical graph for callers of the retired loop entrypoint."""
    from agent_first_browse.agent.graph import run_brain

    return await run_brain(objective)


def main() -> None:
    """Forward legacy command-line usage to the canonical CLI."""
    from agent_first_browse.cli import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    main()
