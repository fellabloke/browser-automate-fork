"""Canonical browser mechanics and session lifecycle."""

from .runtime import SessionGuard, launch_browser, manual_login_mode, shutdown_browser

__all__ = ["SessionGuard", "launch_browser", "manual_login_mode", "shutdown_browser"]
