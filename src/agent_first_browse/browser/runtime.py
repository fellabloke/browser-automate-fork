"""Canonical browser/session lifecycle for the current runtime.

This module owns the launch, attach, persistence, manual-login, and shutdown
behavior previously embedded in ``advanced_agent.py``.  The implementation is
kept intentionally close to the legacy path during migration.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import sys
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright

from agent_first_browse.logging import get_logger
from agent_first_browse.perception import dom as dom_parser
from agent_first_browse.promotion.browser_promoter.cdp_stealth_launcher import (
    STEALTH_INIT_SCRIPT,
    STEALTH_LAUNCH_ARGS,
    STEALTH_USER_AGENT,
    VISUAL_CURSOR_INIT_SCRIPT,
    apply_page_stealth,
    get_random_viewport,
    get_stealth_init_script,
)
from agent_first_browse.browser.site_customizations import (
    apply_current_site_customizations,
    install_site_customizations,
)

logger = get_logger("browser.runtime")

PERSISTENCE_ROOT = Path(__file__).resolve().parents[3] / "persistence"
PROFILE_DIR = PERSISTENCE_ROOT / "browser_sessions" / "agent_main"
_ACTIVE_PLAYWRIGHT = None
_BROWSER_CONNECTION_MODE = "UNINITIALIZED"


def _ensure_dirs():
    PERSISTENCE_ROOT.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def _mark_profile_clean_exit() -> None:
    """Mark the persistent Chrome profile as clean before every launch."""
    prefs_path = PROFILE_DIR / "Default" / "Preferences"
    try:
        if prefs_path.exists():
            data = json.loads(prefs_path.read_text(encoding="utf-8"))
        else:
            prefs_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
        profile = data.setdefault("profile", {})
        profile["exit_type"] = "Normal"
        profile["exited_cleanly"] = True
        prefs_path.write_text(json.dumps(data), encoding="utf-8")
        logger.info("Profile marked clean-exit (crash-restore bubble suppressed)")
    except Exception as e:
        logger.warning("Could not mark profile clean-exit (non-fatal): %s", e)
class SessionGuard:
    """Lightweight lifecycle guard for the browser context.

    With native user_data_dir persistence, Chromium handles all cookie/storage
    persistence internally. This guard only ensures the browser context is
    closed cleanly on exit (which flushes pending state to disk).
    """

    _instance: "SessionGuard | None" = None

    def __init__(self):
        self._context: BrowserContext | None = None
        self._installed = False

    @classmethod
    def get(cls) -> "SessionGuard":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def attach(self, context: BrowserContext):
        """Attach a live browser context to guard."""
        self._context = context
        if not self._installed:
            self._install_handlers()
            self._installed = True
        logger.info("SessionGuard: attached to browser context")

    def _install_handlers(self):
        """Install signal + atexit handlers (once)."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass
        atexit.register(self._atexit_handler)
        logger.info("SessionGuard: signal + atexit handlers installed")

    def _signal_handler(self, signum, frame):
        """Called on Ctrl+C or kill — close context then exit."""
        sig_name = signal.Signals(signum).name
        logger.warning("SessionGuard: caught %s — closing browser...", sig_name)
        # Context.close() flushes all native state to user_data_dir
        self.detach()
        sys.exit(0)

    def _atexit_handler(self):
        """Called on normal interpreter shutdown."""
        if self._context is not None:
            logger.info("SessionGuard: atexit — detaching context")
            self.detach()

    def detach(self):
        """Detach the context reference (called after context.close())."""
        self._context = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Browser Launch — Pure Native Persistence (current)
# ═══════════════════════════════════════════════════════════════════════════════


async def launch_browser(*, headless: bool = False) -> tuple[BrowserContext, Page]:
    """
    Connect to an already-running native Chrome via CDP when LOCAL_CDP_ENDPOINT
    is configured. Otherwise fall back to the existing local Playwright browser.
    """
    global _ACTIVE_PLAYWRIGHT, _BROWSER_CONNECTION_MODE

    _ensure_dirs()

    pw = await async_playwright().start()
    _ACTIVE_PLAYWRIGHT = pw

    cdp_endpoint = os.getenv("LOCAL_CDP_ENDPOINT", "").strip()

    # ============================================================
    # WINDOWS NATIVE CHROME VIA CDP
    # ============================================================
    if cdp_endpoint:
        _BROWSER_CONNECTION_MODE = "LOCAL_CDP"
        logger.info("🌐 Browser connection mode: LOCAL_CDP")
        logger.info("🔗 Attaching to existing Chrome via CDP: %s", cdp_endpoint)

        try:
            browser = await pw.chromium.connect_over_cdp(
                cdp_endpoint,
                timeout=15_000,
            )
        except Exception as e:
            logger.error(
                "❌ Playwright attachment failed for Chrome CDP at %s: %s",
                cdp_endpoint,
                e,
            )
            logger.error(
                "If Chrome is on Windows and Python is in WSL, enable WSL mirrored "
                "networking so 127.0.0.1 is shared."
            )
            await pw.stop()
            _ACTIVE_PLAYWRIGHT = None
            _BROWSER_CONNECTION_MODE = "UNINITIALIZED"
            raise

        if not browser.contexts:
            logger.error("❌ CDP Chrome connected, but no browser context exists.")
            await pw.stop()
            _ACTIVE_PLAYWRIGHT = None
            _BROWSER_CONNECTION_MODE = "UNINITIALIZED"
            raise RuntimeError("Chrome CDP connected but returned no browser contexts.")

        context = browser.contexts[0]

        if context.pages:
            page = context.pages[0]
        else:
            page = await context.new_page()

        # Chrome is hosted by Windows even though this Python process is in WSL.
        # Derive browser-facing fingerprint values from Chrome itself, not Python.
        try:
            browser_platform = await page.evaluate(
                "navigator.userAgentData?.platform || navigator.platform || ''"
            )
        except Exception:
            browser_platform = os.getenv("BROWSER_OS", "Windows")
        logger.info("🖥️ Attached browser reports platform: %s", browser_platform or "unknown")

        # These are context/page instrumentation only; launch-only options such
        # as viewport, locale, timezone, UA and Chromium args are intentionally
        # not supplied to connect_over_cdp().
        await context.add_init_script(
            get_stealth_init_script(browser_platform=str(browser_platform))
        )
        await context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
        await install_site_customizations(context, page)
        await dom_parser.install_shadow_piercer(context)

        try:
            await apply_page_stealth(page)
        except Exception as e:
            logger.warning("Page stealth setup failed (non-fatal): %s", e)

        guard = SessionGuard.get()
        guard.attach(context)

        try:
            await page.bring_to_front()
        except Exception:
            pass

        try:
            title = await page.title()
        except Exception:
            title = ""
        logger.info("✅ CDP attachment ready: url=%s title=%s", page.url, title)

        return context, page

    # ============================================================
    # EXISTING LOCAL WSL CHROMIUM FALLBACK
    # ============================================================
    _mark_profile_clean_exit()
    _BROWSER_CONNECTION_MODE = "LOCAL_PLAYWRIGHT"

    logger.info("🌐 Browser connection mode: LOCAL_PLAYWRIGHT fallback")
    logger.info("Launching local Playwright Chromium (native profile: %s)", PROFILE_DIR)

    session_viewport = get_random_viewport()
    logger.info(
        "Session viewport: %dx%d",
        session_viewport["width"],
        session_viewport["height"],
    )

    try:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=headless,
            viewport=session_viewport,
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=STEALTH_USER_AGENT,
            java_script_enabled=True,
            device_scale_factor=1,
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            args=STEALTH_LAUNCH_ARGS + dom_parser.TLS_STEALTH_ARGS,
            ignore_default_args=["--enable-automation"],
        )
    except Exception as e:
        logger.error("❌ Local Playwright browser launch failed: %s", e)
        await pw.stop()
        _ACTIVE_PLAYWRIGHT = None
        _BROWSER_CONNECTION_MODE = "UNINITIALIZED"
        raise

    await context.add_init_script(STEALTH_INIT_SCRIPT)
    await context.add_init_script(VISUAL_CURSOR_INIT_SCRIPT)
    await install_site_customizations(context)
    await dom_parser.install_shadow_piercer(context)

    guard = SessionGuard.get()
    guard.attach(context)

    if context.pages:
        page = context.pages[0]
    else:
        page = await context.new_page()

    await apply_page_stealth(page)
    await apply_current_site_customizations(page)

    try:
        await page.bring_to_front()
    except Exception:
        pass

    return context, page


async def shutdown_browser(context: BrowserContext) -> None:
    """Release Playwright without terminating an externally managed CDP Chrome."""
    global _ACTIVE_PLAYWRIGHT, _BROWSER_CONNECTION_MODE

    mode = _BROWSER_CONNECTION_MODE
    try:
        if mode == "LOCAL_CDP":
            # The PowerShell launcher owns the dedicated Windows Chrome. Stopping
            # Playwright detaches its CDP transport while leaving Chrome/profile
            # alive for reuse by the next run.
            if _ACTIVE_PLAYWRIGHT is not None:
                await _ACTIVE_PLAYWRIGHT.stop()
            logger.info("Disconnected from Windows Chrome; automation Chrome remains running.")
        else:
            await context.close()
            if _ACTIVE_PLAYWRIGHT is not None:
                await _ACTIVE_PLAYWRIGHT.stop()
            logger.info("Local Playwright browser closed. Profile state persisted.")
    finally:
        _ACTIVE_PLAYWRIGHT = None
        _BROWSER_CONNECTION_MODE = "UNINITIALIZED"


# ═══════════════════════════════════════════════════════════════════════════════
#  Manual Login Mode (--login)
# ═══════════════════════════════════════════════════════════════════════════════
async def manual_login_mode():
    """Open a browser for human login. Chromium natively persists the session."""
    context, page = await launch_browser()

    print("\n" + "=" * 70)
    print("  🔐  MANUAL LOGIN MODE (Native Persistence)")
    print("=" * 70)
    print("  A Chromium browser window has opened.")
    print("  Please log into your Google/Gmail/Reddit account(s) now.")
    print("")
    print("  Once you are fully logged in and can see your inbox/feed,")
    print("  come back here and press ENTER to close the browser.")
    print("  Your session is saved AUTOMATICALLY in the browser profile.")
    print("=" * 70)

    try:
        input("\n  👉 Press ENTER when login is complete: ")
    except (EOFError, KeyboardInterrupt):
        logger.info("Input interrupted — closing browser...")

    try:
        await shutdown_browser(context)
    except Exception:
        pass
    SessionGuard.get().detach()

    print("\n  ✅ Session persisted! Future runs will use your login automatically.\n")
