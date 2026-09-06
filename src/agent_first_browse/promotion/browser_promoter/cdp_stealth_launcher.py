"""Compatibility exports for canonical browser stealth helpers."""

from agent_first_browse.browser.stealth import (
    STEALTH_INIT_SCRIPT,
    STEALTH_LAUNCH_ARGS,
    STEALTH_USER_AGENT,
    VISUAL_CURSOR_INIT_SCRIPT,
    apply_page_stealth,
    get_random_viewport,
    get_stealth_init_script,
    get_stealth_user_agent,
    launch_stealth_context,
)

__all__ = [
    "STEALTH_INIT_SCRIPT",
    "STEALTH_LAUNCH_ARGS",
    "STEALTH_USER_AGENT",
    "VISUAL_CURSOR_INIT_SCRIPT",
    "apply_page_stealth",
    "get_random_viewport",
    "get_stealth_init_script",
    "get_stealth_user_agent",
    "launch_stealth_context",
]
