"""MCP Tool Wrappers — Model Context Protocol interface to existing browser primitives.

Wraps the battle-tested execution layer (cdp_click, cdp_input, ghost_input,
dom_parser, overlay_detector) as MCP-compatible tool functions.

These tools are the ONLY way worker nodes interact with the browser.
They never commit state — they return results that Overwatch validates.

Design:
  - Each tool delegates to an existing, proven module
  - Tool results are structured dicts, not raw strings
  - Page reference is injected at graph construction time
  - All tools are async for Playwright compatibility

References:
  - MCP (Anthropic, Nov 2024): standardized agent-to-tool protocol
  - Playwright MCP: accessibility-tree-first interaction pattern
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import base64
import logging
import os
import random
import re
from typing import Any
from urllib.parse import urlparse

try:
    from agent_first_browse.logging import get_logger
    logger = get_logger("mcp_tools")
except ImportError:
    logger = logging.getLogger("mcp_tools")

# ═══════════════════════════════════════════════════════════════════════════════
#  Tool Registry — holds the live page reference
# ═══════════════════════════════════════════════════════════════════════════════

_PAGE = None  # Set by brain_graph.py at startup
_SELECTOR_MAP: dict[str, dict] = {}
_KNOWN_PAGE_IDS: set[int] = set()
_CRASHED_PAGE_IDS: set[int] = set()
_WATCHED_PAGE_IDS: set[int] = set()


def is_blacklisted_survey_url(url: str) -> bool:
    """Return whether a survey destination must never remain open."""
    raw = os.getenv("SURVEY_BLACKLISTED_URLS", "https://welcome.eu.walr.com")
    host = (urlparse(str(url or "")).hostname or "").lower().removeprefix("www.")
    if not host:
        return False
    for item in re.split(r"[,;\n]+", raw):
        blocked_host = (urlparse(item.strip()).hostname or "").lower().removeprefix("www.")
        if blocked_host and (host == blocked_host or host.endswith("." + blocked_host)):
            return True
    return False


def is_page_crash_error(error: Any) -> bool:
    message = str(error or "").lower()
    return any(marker in message for marker in (
        "target crashed", "page crashed", "target closed", "page has been closed",
        "browser has been closed", "browser context has been closed",
    ))


def _mark_page_crashed(page) -> None:
    if page is not None:
        _CRASHED_PAGE_IDS.add(id(page))


def _watch_page(page) -> None:
    if page is None or id(page) in _WATCHED_PAGE_IDS:
        return
    _WATCHED_PAGE_IDS.add(id(page))
    try:
        page.on("crash", lambda *_args: _mark_page_crashed(page))
    except Exception:
        pass


def set_page(page) -> None:
    """Inject the live Playwright page reference (called once at startup)."""
    global _PAGE, _KNOWN_PAGE_IDS
    _PAGE = page
    try:
        _KNOWN_PAGE_IDS = {id(p) for p in page.context.pages}
        for candidate in page.context.pages:
            _watch_page(candidate)
    except Exception:
        _KNOWN_PAGE_IDS = {id(page)} if page is not None else set()
        _watch_page(page)


def set_selector_map(smap: dict) -> None:
    """Update the current selector map (called each perception cycle)."""
    global _SELECTOR_MAP
    _SELECTOR_MAP = smap


def _get_page():
    if _PAGE is None:
        raise RuntimeError("MCP tools not initialized — call set_page() first")
    return _PAGE


async def verify_action_target(element_id: str | None, verb: str) -> dict[str, Any]:
    """Cheap, side-effect-free target check used before visual escalation."""
    if not element_id:
        return {"ok": False, "reason": "no_dom_element_id"}
    try:
        from agent_first_browse.perception import dom as dom_parser
        result = await dom_parser.resolve_element(_get_page(), element_id)
        if not result.get("ok"):
            return result
        tag = str(result.get("tag") or "").upper()
        role = str(result.get("role") or "").lower()
        if verb == "type":
            valid = tag in {"INPUT", "TEXTAREA"} or role in {"textbox", "searchbox", "combobox"}
        elif verb == "click":
            valid = tag in {"BUTTON", "A", "INPUT", "SELECT", "LABEL", "SUMMARY"} or bool(role)
        else:
            valid = True
        return {**result, "ok": bool(valid),
                "reason": "dom_target_verified" if valid else f"not_actionable_for_{verb}"}
    except Exception as exc:  # pragma: no cover - browser failures are non-fatal
        return {"ok": False, "reason": str(exc)[:120]}


def get_page():
    """Return the page currently owned by the MCP interaction layer."""
    return _get_page()


async def mcp_close_qmee_svg_popup() -> dict:
    """Dismiss the known Qmee privacy/feedback popup close control.

    Qmee renders this control as an unlabeled SVG path, sometimes inside a
    cross-origin survey frame, so it is absent from the accessibility map. The
    exact path signature keeps this preflight narrowly scoped and avoids
    guessing at arbitrary page coordinates.
    """
    page = _get_page()
    selector = (
        'path[stroke="#3C3C3C"]'
        '[d="M13.368 12.629 8.183 7.814 13.368 3M2.999 3l5.184 4.815L3 12.629"]'
    )
    try:
        for frame in page.frames:
            path = frame.locator(selector)
            count = await path.count()
            for index in range(count):
                candidate = path.nth(index)
                if not await candidate.is_visible():
                    continue
                result = await candidate.evaluate("""
                (node) => {
                    let target = node;
                    for (let hops = 0; target && hops < 7; hops++, target = target.parentElement) {
                        if (target.matches && target.matches(
                            'button,a,[role="button"],[role="link"],[onclick],[tabindex]'
                        )) break;
                    }
                    target = target || node;
                    target.dispatchEvent(new MouseEvent('click', {
                        bubbles: true, cancelable: true, view: window,
                    }));
                    return {tag: target.tagName || '', className: String(target.className || '').slice(0, 100)};
                }
                """)
                await asyncio.sleep(0.15)
                remaining = await frame.locator(selector).count()
                logger.info(
                    "🪟 Qmee SVG popup close attempted in frame=%s target=%s remaining_paths=%d",
                    str(getattr(frame, "url", ""))[:120], result.get("tag", "?"), remaining,
                )
                if remaining == 0:
                    return {"closed": True, "success": True}
        return {"closed": False, "success": True}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Qmee SVG popup preflight skipped: %s", str(exc)[:160])
        return {"closed": False, "success": False, "error": str(exc)[:160]}


async def recover_unusable_page(
    current_page=None,
    *,
    fallback_url: str = "",
    force: bool = False,
) -> dict:
    """Replace a crashed/closed renderer while preserving its browser context.

    A renderer crash is not a mission-ending condition. Prefer an existing tab
    on the requested provider; otherwise create a clean target and navigate it
    to the provider dashboard. Unrelated background tabs are never hijacked.
    """
    global _PAGE, _SELECTOR_MAP, _KNOWN_PAGE_IDS
    current = current_page or _PAGE
    if current is None:
        return {"recovered": False, "page": None, "reason": "no active page"}
    try:
        closed = current.is_closed()
    except Exception:
        closed = True
    if not force and not closed and id(current) not in _CRASHED_PAGE_IDS:
        return {"recovered": False, "page": current, "reason": "page is usable"}

    try:
        context = current.context
        pages = list(context.pages)
        fallback_host = urlparse(fallback_url).hostname or ""
        candidates = []
        for candidate in pages:
            if candidate is current or id(candidate) in _CRASHED_PAGE_IDS:
                continue
            try:
                if candidate.is_closed():
                    continue
            except Exception:
                continue
            if fallback_host and fallback_host not in (getattr(candidate, "url", "") or ""):
                continue
            candidates.append(candidate)
        replacement = candidates[-1] if candidates else await context.new_page()
        _watch_page(replacement)
        if fallback_url and (getattr(replacement, "url", "") or "") != fallback_url:
            await replacement.goto(
                fallback_url, wait_until="domcontentloaded", timeout=20_000
            )
        try:
            await replacement.bring_to_front()
        except Exception:
            pass
        _PAGE = replacement
        _SELECTOR_MAP = {}
        _KNOWN_PAGE_IDS.update(id(page) for page in context.pages)
        logger.warning(
            "🛟 Recovered unusable browser target%s",
            f" at {fallback_url[:100]}" if fallback_url else "",
        )
        return {"recovered": True, "page": replacement, "reason": "fresh browser target"}
    except Exception as exc:
        logger.error("🛟 Browser target recovery failed: %s", exc)
        return {"recovered": False, "page": current, "reason": str(exc)[:180]}


async def adopt_new_page_if_opened(current_page=None, wait_ms: int = 2500) -> dict:
    """Adopt a newly opened tab/popup, or recover when the active tab closes.

    Browser actions often launch their real destination in a new tab while the
    opener re-renders in place. Without an explicit handoff, perception remains
    attached to the opener and incorrectly concludes that the destination never
    started. New pages whose opener is the current page are preferred; otherwise
    the newest live page wins. Existing background tabs are never selected.
    """
    global _PAGE, _KNOWN_PAGE_IDS

    current = current_page or _PAGE
    if current is None:
        return {"switched": False, "page": None, "reason": "no active page"}

    try:
        context = current.context
        pages = list(context.pages)
    except Exception as exc:
        return {"switched": False, "page": current, "reason": str(exc)[:120]}

    live_pages = []
    for candidate in pages:
        _watch_page(candidate)
        try:
            if not candidate.is_closed():
                live_pages.append(candidate)
        except Exception:
            live_pages.append(candidate)

    new_pages = [p for p in live_pages if id(p) not in _KNOWN_PAGE_IDS]
    _KNOWN_PAGE_IDS.update(id(p) for p in pages)

    try:
        current_closed = current.is_closed()
    except Exception:
        current_closed = current not in live_pages

    # A provider can navigate the existing tab directly to a disallowed
    # redirect instead of opening a popup. Close it before perception sees it.
    if not current_closed and is_blacklisted_survey_url(getattr(current, "url", "")):
        replacement_candidates = [
            p for p in live_pages
            if p is not current and not is_blacklisted_survey_url(getattr(p, "url", ""))
        ]
        replacement = replacement_candidates[-1] if replacement_candidates else await context.new_page()
        try:
            await current.close()
        except Exception as exc:
            logger.debug("Blacklisted survey tab close skipped: %s", exc)
        _PAGE = replacement
        await replacement.bring_to_front()
        logger.warning("🚫 Closed blacklisted survey URL: %s", getattr(current, "url", "")[:160])
        return {
            "switched": True,
            "page": replacement,
            "old_url": getattr(current, "url", ""),
            "new_url": getattr(replacement, "url", "about:blank"),
            "blocked_url": getattr(current, "url", ""),
            "reason": "blacklisted_url",
        }

    if not new_pages and not current_closed:
        # A previous post-action handoff may already have updated the MCP page.
        return {"switched": _PAGE is not current, "page": _PAGE or current,
                "reason": "already adopted" if _PAGE is not current else "no new page"}

    candidates = new_pages or [p for p in live_pages if p is not current]
    if not candidates:
        return {"switched": False, "page": current, "reason": "no live replacement"}

    direct_popups = []
    for candidate in candidates:
        try:
            if await candidate.opener() is current:
                direct_popups.append(candidate)
        except Exception:
            pass
    selected = (direct_popups or candidates)[-1]

    # Popup creation commonly precedes its navigation. Give the selected page a
    # short bounded window to leave about:blank and build an inspectable DOM.
    deadline = asyncio.get_running_loop().time() + max(0, wait_ms) / 1000.0
    while getattr(selected, "url", "") in ("", "about:blank"):
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.05)
    try:
        remaining_ms = max(100, int((deadline - asyncio.get_running_loop().time()) * 1000))
        await selected.wait_for_load_state("domcontentloaded", timeout=remaining_ms)
    except Exception:
        pass

    if is_blacklisted_survey_url(getattr(selected, "url", "")):
        blocked_url = getattr(selected, "url", "")
        try:
            await selected.close()
        except Exception as exc:
            logger.debug("Blacklisted popup close skipped: %s", exc)
        replacement = current
        try:
            if replacement.is_closed() or is_blacklisted_survey_url(getattr(replacement, "url", "")):
                replacement = next(
                    (
                        p for p in live_pages
                        if not p.is_closed()
                        and not is_blacklisted_survey_url(getattr(p, "url", ""))
                    ),
                    None,
                )
                if replacement is None:
                    replacement = await context.new_page()
        except Exception:
            replacement = await context.new_page()
        _PAGE = replacement
        try:
            await replacement.bring_to_front()
        except Exception:
            pass
        logger.warning("🚫 Closed blacklisted survey popup: %s", blocked_url[:160])
        return {
            "switched": replacement is not current,
            "page": replacement,
            "old_url": old_url if "old_url" in locals() else getattr(current, "url", ""),
            "new_url": getattr(replacement, "url", "about:blank"),
            "blocked_url": blocked_url,
            "reason": "blacklisted_url",
        }
    try:
        await selected.bring_to_front()
    except Exception:
        pass

    old_url = getattr(current, "url", "")
    new_url = getattr(selected, "url", "")
    _PAGE = selected
    logger.info(
        "🗂️ TAB HANDOFF: %s → %s (%s)",
        old_url[:100] or "about:blank",
        new_url[:100] or "about:blank",
        "new popup" if new_pages else "active tab closed",
    )
    return {
        "switched": selected is not current,
        "page": selected,
        "old_url": old_url,
        "new_url": new_url,
        "reason": "new_popup" if new_pages else "active_closed",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Navigate
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_navigate(url: str) -> dict:
    """Navigate the browser to a URL.

    Delegates to: page.goto()
    Returns: {"success": bool, "url": str, "error": str}
    """
    global _PAGE, _KNOWN_PAGE_IDS
    page = _get_page()
    try:
        if is_blacklisted_survey_url(url):
            logger.warning("🚫 Blocked navigation to blacklisted survey URL: %s", url[:160])
            return {"success": False, "url": page.url, "error": f"Blacklisted survey URL: {url[:120]}"}
        # Domain safety check (existing module)
        from agent_first_browse.verification.safety import is_domain_allowed
        if not is_domain_allowed(url):
            return {"success": False, "url": page.url, "error": f"Domain blocked: {url[:80]}"}

        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        # A fresh page resets the real mouse to (0,0) and the visual cursor is
        # hidden — re-place the arrow at the tracked position so it never flashes
        # at the top-left corner (which would mislead a vision screenshot).
        try:
            from agent_first_browse.browser.ghost_input import resync_visual_cursor
            await resync_visual_cursor(page)
        except Exception:
            pass
        return {"success": True, "url": page.url, "error": ""}
    except Exception as e:
        logger.warning("mcp_navigate failed: %s", e)
        error_text = str(e)[:200]
        # A third-party survey tab can crash its renderer/iframe. A crashed
        # Playwright Page cannot be navigated again, so recover through a live
        # dashboard/opener tab or create a fresh tab in the same context.
        if is_page_crash_error(error_text):
            _mark_page_crashed(page)
            recovery = await recover_unusable_page(
                page, fallback_url=url, force=True
            )
            if recovery.get("recovered"):
                replacement = recovery["page"]
                return {"success": True, "url": replacement.url, "error": ""}
            return {"success": False, "url": url, "error": error_text}
        try:
            current_url = page.url
        except Exception:
            current_url = url
        return {"success": False, "url": current_url, "error": error_text}


async def mcp_abandon_survey(url: str, *, fresh_dashboard: bool = False) -> dict:
    """Close the active survey target and restore an ordered provider dashboard.

    A fresh target is created before closing the only live tab, preserving the
    browser context/cookies while ensuring a crashed or wedged questionnaire is
    not reused. When the provider dashboard/opener is already open, it is reused.
    """
    global _PAGE, _KNOWN_PAGE_IDS
    current = _get_page()
    if not url:
        return {"success": False, "url": getattr(current, "url", ""), "error": "Missing provider URL"}
    try:
        from agent_first_browse.verification.safety import is_domain_allowed
        if not is_domain_allowed(url):
            return {
                "success": False,
                "url": getattr(current, "url", ""),
                "error": f"Domain blocked: {url[:80]}",
            }

        context = current.context
        target_host = (urlparse(url).hostname or "").removeprefix("www.")
        live = [page for page in context.pages if not page.is_closed() and page is not current]

        def same_provider(page) -> bool:
            try:
                host = (urlparse(page.url).hostname or "").removeprefix("www.")
                return bool(target_host and (host == target_host or host.endswith("." + target_host)))
            except Exception:
                return False

        matching = [page for page in live if same_provider(page)]
        destination = (
            await context.new_page()
            if fresh_dashboard
            else (matching[-1] if matching else await context.new_page())
        )
        _PAGE = destination
        _KNOWN_PAGE_IDS.update(id(page) for page in context.pages)

        if not fresh_dashboard and current is not destination and not current.is_closed():
            try:
                await current.close()
            except Exception as close_error:
                logger.debug("Survey tab close skipped (non-fatal): %s", close_error)

        await destination.goto(url, wait_until="domcontentloaded", timeout=30000)
        if fresh_dashboard:
            # Qmee completion redirects can leave stale concurrency/session
            # state in both the old survey tab and its dashboard opener. Only
            # after the new authenticated dashboard has loaded do we close them.
            stale_tabs = [current, *matching]
            for stale in stale_tabs:
                if stale is destination:
                    continue
                try:
                    if not stale.is_closed():
                        await stale.close()
                except Exception as close_error:
                    logger.debug("Stale provider tab close skipped (non-fatal): %s", close_error)
        try:
            await destination.bring_to_front()
        except Exception:
            pass
        try:
            from agent_first_browse.browser.ghost_input import resync_visual_cursor
            await resync_visual_cursor(destination)
        except Exception:
            pass
        if fresh_dashboard:
            logger.info("🗂️ Opened fresh provider dashboard and closed stale tabs: %s", destination.url[:120])
        else:
            logger.info("↩️ Closed abandoned survey and restored provider: %s", destination.url[:120])
        return {"success": True, "url": destination.url, "error": ""}
    except Exception as exc:
        logger.warning("mcp_abandon_survey failed: %s", exc)
        try:
            fallback = await mcp_navigate(url)
            if fallback.get("success"):
                return fallback
        except Exception:
            pass
        return {
            "success": False,
            "url": getattr(_PAGE, "url", url),
            "error": str(exc)[:200],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Click
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_click_point(
    rect: dict,
    fallback_x: float,
    fallback_y: float,
) -> tuple[float, float]:
    """Choose a bounded point inside a live target box.

    Live resolution used to force every click to the exact centre. Sampling
    the middle 40% keeps compact controls safe while avoiding a fixed target
    point on repeated interactions.
    """
    try:
        left = float(rect["x"])
        top = float(rect["y"])
        width = float(rect["width"])
        height = float(rect["height"])
        if width < 8 or height < 8:
            return float(fallback_x), float(fallback_y)
        return (
            round(left + width * random.uniform(0.30, 0.70), 2),
            round(top + height * random.uniform(0.30, 0.70), 2),
        )
    except (KeyError, TypeError, ValueError):
        return float(fallback_x), float(fallback_y)


async def _sample_coordinate_target(page, x: float, y: float) -> tuple[float, float]:
    """Sample inside the actionable control currently under raw coordinates.

    Coordinate-only vision actions do not have an element id to resolve.  A
    short, read-only hit test covers ordinary buttons/labels without changing
    the behaviour of coordinates that land on a canvas, iframe, or decorative
    node.
    """
    try:
        target = await asyncio.wait_for(page.evaluate("""
        ({x, y}) => {
            const hit = document.elementFromPoint(x, y);
            if (!hit) return null;
            const target = hit.closest && (
                hit.closest('button,a,label,[role="button"],[role="link"],'
                            + '[role="radio"],[role="checkbox"],[role="option"]')
                || hit
            );
            if (!target || !target.getBoundingClientRect) return null;
            const r = target.getBoundingClientRect();
            if (r.width < 8 || r.height < 8) return null;
            return {x: r.left, y: r.top, width: r.width, height: r.height};
        }
        """, {"x": float(x), "y": float(y)}), timeout=1.5)
        if isinstance(target, dict):
            return _sample_click_point(target, x, y)
    except Exception:
        pass
    return float(x), float(y)

async def mcp_click(
    x: float, y: float,
    element_id: str | None = None,
    prevent_deselect: bool = False,
    replay_safe: bool = False,
) -> dict:
    """Click an element at (x, y) with overlay penetration and CDP resilience.

    Delegates to:
      1. overlay_detector.smart_click_with_penetration()
      2. ghost_input.ghost_move_to() — Bézier curve humanization
      3. cdp_click.resilient_click() — 4-strategy waterfall

    Returns: {"success": bool, "strategy": str, "navigated": bool, "dom_changed": bool, "error": str}
    """
    page = _get_page()

    try:
        # Qmee's close control is an unlabeled SVG path and may be inside the
        # survey iframe, so it can appear after the last snapshot and bypass
        # normal popup-first DOM routing. Clear it before any other click.
        qmee_popup = await mcp_close_qmee_svg_popup()
        if qmee_popup.get("closed"):
            logger.info("🪟 Closed Qmee SVG popup before requested click")
            return {
                "success": True, "strategy": "qmee_svg_popup_close",
                "navigated": False, "dom_changed": True, "verified": True,
                "state_verified": True, "no_op": False, "error": "",
            }
        # Resolve the chosen element to FRESH, identity-verified coords from
        # the exact node the LLM picked (drift-proof). Falls back to snapshot
        # coords if the registry has no live node for this id.
        if element_id:
            from agent_first_browse.perception import dom as dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                center_x, center_y = float(r["x"]), float(r["y"])
                x, y = _sample_click_point(r.get("rect") or {}, center_x, center_y)
                logger.info("🎯 Resolved %s → fresh target (%d,%d) from center (%d,%d) [%s '%s']",
                            element_id, int(x), int(y), int(center_x), int(center_y), r.get("tag", ""),
                            (r.get("text", "") or "")[:25])
                await asyncio.sleep(random.uniform(0.04, 0.11))
            else:
                logger.debug("resolve %s miss (%s) — using snapshot coords",
                             element_id, r.get("reason"))
        else:
            sampled_x, sampled_y = await _sample_coordinate_target(page, x, y)
            if (sampled_x, sampled_y) != (float(x), float(y)):
                logger.info("🎯 Raw coordinate target (%d,%d) → sampled (%d,%d)",
                            int(x), int(y), int(sampled_x), int(sampled_y))
                x, y = sampled_x, sampled_y

        # Step 1: Overlay penetration
        from agent_first_browse.browser.overlays import smart_click_with_penetration
        penetration = await smart_click_with_penetration(page, x, y)
        if penetration.get("overlay_bypassed"):
            logger.info("🎯 Overlay penetrated at (%d, %d) via %s", x, y, penetration.get("method", "?"))
            await asyncio.sleep(0.04)

        # Step 2: Humanized mouse movement
        from agent_first_browse.browser.ghost_input import ghost_move_to
        await ghost_move_to(page, x, y)

        # Step 3: CDP resilient click
        from agent_first_browse.browser.cdp_click import resilient_click
        click_result = await asyncio.wait_for(
            resilient_click(
                page, x, y, max_retries=3,
                settle_ms=max(200, int(os.getenv("CLICK_SETTLE_MAX_MS", "600"))),
                element_id=element_id,
                prevent_deselect=prevent_deselect,
                replay_safe=replay_safe,
            ),
            timeout=30.0,
        )

        if click_result.success:
            return {
                "success": True,
                "strategy": click_result.strategy,
                "navigated": click_result.navigation,
                "dom_changed": click_result.dom_changed,
                "verified": click_result.verified,
                "state_verified": "state_verified" in click_result.strategy,
                "no_op": bool(getattr(click_result, "no_op", False)),
                "error": "",
            }
        else:
            # ── current SAFETY NET: element-id state verification ──────────────
            # If the coordinate-based Visual Truth Override in cdp_click.py
            # missed the state change (e.g. elementFromPoint returned the
            # label instead of the input), use the element_id registry to
            # check the actual element's state directly.
            # ───────────────────────────────────────────────────────────────
            if element_id:
                try:
                    state_check = await page.evaluate("""
                    (eid) => {
                        const el = document.querySelector(`[__aid="${eid}"]`);
                        if (!el) return {found: false};
                        return {
                            found: true,
                            checked: !!el.checked,
                            selected: !!el.selected,
                            value: (el.value || '').slice(0, 50),
                            ariaChecked: el.getAttribute('aria-checked') || '',
                        };
                    }
                    """, element_id)
                    if state_check.get("found") and (
                        state_check.get("checked") or
                        state_check.get("ariaChecked") == "true"
                    ):
                        logger.info(
                            "✅ current SAFETY NET: click engine reported failure but "
                            "element %s state is checked=%s. Overriding to success.",
                            element_id, state_check.get("checked"),
                        )
                        return {
                            "success": True,
                            "strategy": "element_state_verified",
                            "navigated": False,
                            "dom_changed": False,
                            "error": "",
                        }
                except Exception as e:
                    logger.debug("current safety net check failed: %s", e)

            return {
                "success": False,
                "strategy": "",
                "navigated": False,
                "dom_changed": False,
                "error": f"Click ineffective: {click_result.error[:80]}",
            }

    except asyncio.TimeoutError:
        return {"success": False, "strategy": "", "navigated": False,
                "dom_changed": False, "error": "Click timed out (30s)"}
    except Exception as e:
        logger.warning("mcp_click failed: %s", e)
        if is_page_crash_error(e):
            _mark_page_crashed(page)
        return {"success": False, "strategy": "", "navigated": False,
                "dom_changed": False, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Type
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_type(
    text: str,
    x: float, y: float,
    element_id: str | None = None,
    clear_first: bool = True,
    force_retype: bool = False,
) -> dict:
    """Type text into an element with CDP resilient typing.

    Delegates to: cdp_input.resilient_type()
    Returns: {"success": bool, "strategy": str, "actual_length": int, "error": str}
    """
    page = _get_page()

    try:
        typing_element_id = None
        # Resolve to fresh, identity-verified coords from the chosen node.
        if element_id:
            from agent_first_browse.perception import dom as dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                x, y = float(r["x"]), float(r["y"])
                typing_element_id = r.get("resolved_id") or (
                    None if str(r.get("requested_tag", "")).upper() == "LABEL" else element_id
                )
                logger.info("🎯 Resolved %s → fresh (%d,%d) [%s '%s']",
                            element_id, int(x), int(y), r.get("tag", ""),
                            (r.get("text", "") or "")[:25])
                await asyncio.sleep(0.2)  # let scroll settle
            else:
                typing_element_id = element_id
                logger.debug("resolve %s miss (%s) — using snapshot coords",
                             element_id, r.get("reason"))

        from agent_first_browse.browser.cdp_input import resilient_type
        type_result = await asyncio.wait_for(
            resilient_type(
                page, text, x=x, y=y,
                clear_first=clear_first, force_retype=force_retype, max_retries=3,
                element_id=typing_element_id,
            ),
            timeout=60.0,
        )

        if type_result["success"]:
            return {
                "success": True,
                "strategy": type_result["strategy"],
                "actual_length": type_result["actual_length"],
                "no_op": bool(type_result.get("no_op")),
                "error": "",
            }
        else:
            return {
                "success": False,
                "strategy": "",
                "actual_length": 0,
                "error": "Type unverified after all strategies",
            }

    except asyncio.TimeoutError:
        return {"success": False, "strategy": "", "actual_length": 0, "error": "Type timed out (60s)"}
    except Exception as e:
        logger.warning("mcp_type failed: %s", e)
        return {"success": False, "strategy": "", "actual_length": 0, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Scroll
# ═══════════════════════════════════════════════════════════════════════════════

_SCROLL_METRICS_JS = """
() => {
  const se = document.scrollingElement || document.documentElement || document.body;
  return {
    y: Math.round(window.scrollY || (se && se.scrollTop) || 0),
    h: Math.round((se && se.scrollHeight) || (document.body && document.body.scrollHeight) || 0),
    vh: Math.round(window.innerHeight || (se && se.clientHeight) || 0),
  };
}
"""


async def mcp_scroll(pixels: int = 600) -> dict:
    """Scroll the page down WITH feedback (current smart scroll).

    Delegates the humanized scroll to ghost_input.ghost_scroll (FROZEN — unchanged)
    and only MEASURES the viewport before/after, so the brain knows whether the page
    actually moved and whether it has reached the bottom. This is what lets the agent
    stop scrolling into a wall instead of looping.

    Returns: {success, error, scrolled_px, at_bottom, position, page_height, viewport_h}
    """
    page = _get_page()
    try:
        try:
            before = await page.evaluate(_SCROLL_METRICS_JS)
        except Exception:
            before = {}
        from agent_first_browse.browser.ghost_input import ghost_scroll
        await asyncio.wait_for(ghost_scroll(page, pixels), timeout=10.0)
        try:
            after = await page.evaluate(_SCROLL_METRICS_JS)
        except Exception:
            after = {}
        by = int(before.get("y", 0) or 0)
        ay = int(after.get("y", by) or by)
        ph = int(after.get("h", before.get("h", 0)) or 0)
        vh = int(after.get("vh", before.get("vh", 0)) or 0)
        at_bottom = bool(ph and vh and (ay + vh >= ph - 4))
        return {"success": True, "error": "", "scrolled_px": ay - by,
                "at_bottom": at_bottom, "position": ay,
                "page_height": ph, "viewport_h": vh}
    except asyncio.TimeoutError:
        return {"success": False, "error": "Scroll timed out (10s)",
                "scrolled_px": 0, "at_bottom": False}
    except Exception as e:
        logger.warning("mcp_scroll failed: %s", e)
        return {"success": False, "error": str(e)[:200],
                "scrolled_px": 0, "at_bottom": False}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Press Enter
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_press_enter() -> dict:
    """Press the Enter key.

    Returns: {"success": bool, "error": str}
    """
    page = _get_page()
    try:
        await page.keyboard.press("Enter")
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_press_enter failed: %s", e)
        return {"success": False, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  A — Expanded primitives (hover / select_option / press_key)
#  Universal handlers reusing existing backends (ghost_move_to, keyboard, __aid).
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_press_key(key: str) -> dict:
    """Press a single key or chord (e.g. 'Escape', 'Tab', 'ArrowDown', 'Control+A').
    Reuses Playwright keyboard. Universal — Playwright validates the key name."""
    page = _get_page()
    key = (key or "").strip()
    if not key or len(key) > 40:
        return {"success": False, "error": "no/invalid key specified"}
    try:
        await page.keyboard.press(key)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_press_key failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


async def mcp_hover(element_id: str | None = None, x: float = 0, y: float = 0) -> dict:
    """Hover over an element (reveal menus/tooltips/submenus). Resolves fresh
    coordinates via the element registry, then reuses the humanized ghost_move_to."""
    page = _get_page()
    try:
        if element_id:
            from agent_first_browse.perception import dom as dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                x, y = float(r["x"]), float(r["y"])
        from agent_first_browse.browser.ghost_input import ghost_move_to
        await ghost_move_to(page, x, y)
        await asyncio.sleep(0.15)  # let hover-triggered UI settle
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_hover failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


_SELECT_OPTION_JS = r"""
(args) => {
  const el = window.__aid && window.__aid[args.id];
  if (!el) return {ok:false, reason:'no live node for id'};
  if ((el.tagName||'').toUpperCase() !== 'SELECT') return {ok:false, reason:'not a native select element'};
  const want = String(args.value||''), wl = want.toLowerCase();
  let m = null;
  for (const o of el.options) { if (o.value === want || (o.textContent||'').trim() === want) { m = o; break; } }
  if (!m) { for (const o of el.options) { if ((o.textContent||'').toLowerCase().includes(wl)) { m = o; break; } } }
  if (!m) return {ok:false, reason:'option not found: '+want.slice(0,40)};
  el.value = m.value;
  el.dispatchEvent(new Event('input', {bubbles:true}));
  el.dispatchEvent(new Event('change', {bubbles:true}));
  return {ok:true, selected:(m.textContent||'').trim().slice(0,40)};
}
"""


async def mcp_select_option(element_id: str | None, value: str) -> dict:
    """Choose an option in a NATIVE <select> by value or visible text (sets the
    value + dispatches input/change). Universal: works on any native dropdown via
    the __aid registry node; returns a clear reason if the target isn't a <select>
    (so the agent falls back to clicking a custom/ARIA dropdown)."""
    page = _get_page()
    if not element_id:
        return {"success": False, "error": "select_option requires an element_id"}
    try:
        res = await page.evaluate(_SELECT_OPTION_JS, {"id": element_id, "value": value or ""})
        if res.get("ok"):
            return {"success": True, "selected": res.get("selected", ""), "error": ""}
        return {"success": False, "error": res.get("reason", "select failed")}
    except Exception as e:
        logger.warning("mcp_select_option failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


_SET_DATE_OF_BIRTH_JS = r"""
(args) => {
  const iso = String(args.iso || '');
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!match) return {ok:false, reason:'invalid ISO date'};
  const year = match[1], month = match[2], day = match[3];
  let target = window.__aid && window.__aid[args.id];
  const visible = el => {
    if (!el) return false;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 1 && r.height > 1 && s.display !== 'none' && s.visibility !== 'hidden';
  };
  const label = el => {
    if (!el) return '';
    let text = [el.getAttribute('aria-label'), el.getAttribute('placeholder'),
      el.getAttribute('name'), el.getAttribute('id'), el.getAttribute('autocomplete')]
      .filter(Boolean).join(' ');
    try {
      const by = el.getAttribute('aria-labelledby') || '';
      text += ' ' + by.split(/\s+/).map(id => document.getElementById(id)?.textContent || '').join(' ');
      if (el.labels) text += ' ' + [...el.labels].map(x => x.textContent || '').join(' ');
    } catch(_) {}
    return text.replace(/\s+/g, ' ').trim().toLowerCase();
  };
  const setNative = (el, value) => {
    try {
      const proto = el.tagName === 'SELECT' ? HTMLSelectElement.prototype
        : el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
      if (setter) setter.call(el, value); else el.value = value;
      el.dispatchEvent(new Event('input', {bubbles:true}));
      el.dispatchEvent(new Event('change', {bubbles:true}));
      el.dispatchEvent(new Event('blur', {bubbles:true}));
      return true;
    } catch(_) { return false; }
  };
  const choose = (select, variants) => {
    const options = [...select.options];
    const norm = x => String(x || '').trim().toLowerCase().replace(/^0+(?=\d)/, '');
    const wanted = variants.map(norm);
    const option = options.find(o => wanted.includes(norm(o.value)) || wanted.includes(norm(o.textContent)));
    return option ? setNative(select, option.value) : false;
  };

  if (target && target.tagName === 'INPUT' && target.type === 'date') {
    setNative(target, iso);
    return target.value === iso ? {ok:true, mode:'native-date'} : {ok:false, reason:'native date rejected value'};
  }

  let scope = target?.closest('fieldset,[role="group"],[data-question-id],[data-question],.question,.form-group') || document;
  let controls = [...scope.querySelectorAll('input,select')].filter(visible);
  let dated = controls.filter(el => /(?:birth|dob|day|month|year|dd|mm|yyyy)/i.test(label(el)));
  if (!dated.length) {
    dated = [...document.querySelectorAll('input[type="date"],input,select')]
      .filter(el => visible(el) && /(?:birth|dob|day|month|year|dd|mm|yyyy)/i.test(label(el)));
  }
  const native = dated.find(el => el.tagName === 'INPUT' && el.type === 'date');
  if (native) {
    setNative(native, iso);
    return native.value === iso ? {ok:true, mode:'native-date'} : {ok:false, reason:'native date rejected value'};
  }

  const allMonths = [
    ['january','jan'],['february','feb'],['march','mar'],['april','apr'],
    ['may','may'],['june','jun'],['july','jul'],['august','aug'],
    ['september','sep'],['october','oct'],['november','nov'],['december','dec']
  ];
  const monthNames = allMonths[Math.max(0, Number(month) - 1)] || [];
  const parts = {day:false, month:false, year:false};
  for (const el of dated) {
    const desc = label(el);
    let part = '';
    if (/\b(?:day|dd)\b/.test(desc)) part = 'day';
    else if (/\b(?:month|mm)\b/.test(desc)) part = 'month';
    else if (/\b(?:year|yyyy)\b/.test(desc)) part = 'year';
    if (!part) continue;
    const variants = part === 'day' ? [day, String(Number(day))]
      : part === 'month' ? [month, String(Number(month)), ...monthNames]
      : [year];
    if (el.tagName === 'SELECT') parts[part] = choose(el, variants);
    else parts[part] = setNative(el, variants[0]);
  }
  if (parts.day && parts.month && parts.year) return {ok:true, mode:'segmented-date'};
  return {ok:false, reason:'could not bind all birthdate components', parts};
}
"""


_VERIFY_DATE_OF_BIRTH_JS = r"""
(args) => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(args.iso || ''));
  if (!match) return {ok:false, reason:'invalid ISO date'};
  const expected = {year:match[1], month:match[2], day:match[3]};
  const target = window.__aid && window.__aid[args.id];
  const visible = el => {
    if (!el || !el.isConnected) return false;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 1 && r.height > 1 && s.display !== 'none' && s.visibility !== 'hidden';
  };
  const label = el => {
    let text = [el.getAttribute('aria-label'), el.getAttribute('placeholder'),
      el.getAttribute('name'), el.getAttribute('id'), el.getAttribute('autocomplete')]
      .filter(Boolean).join(' ');
    try {
      const by = el.getAttribute('aria-labelledby') || '';
      text += ' ' + by.split(/\s+/).map(id => document.getElementById(id)?.textContent || '').join(' ');
      if (el.labels) text += ' ' + [...el.labels].map(x => x.textContent || '').join(' ');
    } catch(_) {}
    return text.replace(/\s+/g, ' ').trim().toLowerCase();
  };
  const native = target?.matches?.('input[type="date"]') ? target
    : [...document.querySelectorAll('input[type="date"]')].find(visible);
  if (native) return native.value === args.iso
    ? {ok:true, mode:'native-date'} : {ok:false, reason:'native date value reverted'};
  const scope = target?.closest?.('fieldset,[role="group"],[data-question-id],[data-question],.question,.form-group') || document;
  let controls = [...scope.querySelectorAll('input,select')].filter(visible);
  if (controls.length < 3) controls = [...document.querySelectorAll('input,select')].filter(visible);
  const actual = {};
  const monthNames = ['january','february','march','april','may','june','july','august','september','october','november','december'];
  for (const el of controls) {
    const desc = label(el);
    const raw = String(el.value || '').trim().toLowerCase();
    if (/\b(?:day|dd)\b/.test(desc)) actual.day = raw.replace(/^0+(?=\d)/, '');
    else if (/\b(?:month|mm)\b/.test(desc)) {
      const monthIndex = monthNames.findIndex(name => raw === name || raw === name.slice(0,3));
      actual.month = monthIndex >= 0 ? String(monthIndex + 1) : raw.replace(/^0+(?=\d)/, '');
    } else if (/\b(?:year|yyyy)\b/.test(desc)) actual.year = raw;
  }
  const ok = actual.day === String(Number(expected.day))
    && actual.month === String(Number(expected.month)) && actual.year === expected.year;
  return ok ? {ok:true, mode:'segmented-date'}
    : {ok:false, reason:'one or more date components reverted'};
}
"""


async def mcp_set_date_of_birth(element_id: str | None, iso_date: str) -> dict:
    """Fill one native/segmented DOB widget, switching to manual calendar when offered."""
    page = _get_page()
    if not element_id:
        return {"success": False, "error": "date of birth action requires an element_id"}
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(iso_date or "")):
        return {"success": False, "error": "date of birth must use YYYY-MM-DD"}
    try:
        # Providers sometimes default to a visual picker and expose a safer
        # segmented/manual alternative. Prefer it before manipulating controls.
        switched = await page.evaluate(r"""() => {
            const candidates = [...document.querySelectorAll('button,a,[role="button"]')];
            const alternative = candidates.find(el => {
                const text = (el.innerText || el.textContent || el.getAttribute('aria-label') || '')
                    .replace(/\s+/g, ' ').trim();
                if (!/(alternative calendar|enter date manually|type date|manual entry|switch calendar)/i.test(text)) return false;
                const r = el.getBoundingClientRect(), s = getComputedStyle(el);
                return r.width > 1 && r.height > 1 && s.display !== 'none' && s.visibility !== 'hidden';
            });
            if (!alternative) return false;
            alternative.click();
            return true;
        }""")
        if switched:
            await page.wait_for_timeout(400)
        result = await page.evaluate(
            _SET_DATE_OF_BIRTH_JS,
            {"id": element_id, "iso": iso_date},
        )
        if not result.get("ok"):
            return {"success": False, "error": str(result.get("reason") or "date widget failed")[:180]}
        # Framework-controlled inputs can revert after the event handler runs.
        # Re-read all components after the render cycle before reporting success.
        await page.wait_for_timeout(250)
        verified = await page.evaluate(
            _VERIFY_DATE_OF_BIRTH_JS,
            {"id": element_id, "iso": iso_date},
        )
        if verified.get("ok"):
            logger.info("📅 Date-of-birth widget completed via %s", verified.get("mode", "profile date"))
            return {"success": True, "mode": verified.get("mode", ""), "error": ""}
        return {"success": False, "error": str(verified.get("reason") or "date verification failed")[:180]}
    except Exception as exc:
        logger.warning("mcp_set_date_of_birth failed: %s", exc)
        return {"success": False, "error": str(exc)[:180]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Comprehensive Browser Action Suite — New Primitives
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_drag_and_drop(
    element_id: str | None = None,
    x: float = 0, y: float = 0,
    target_x: float = 0, target_y: float = 0,
    target_element_id: str | None = None,
) -> dict:
    """Drag from (x, y) to (target_x, target_y) using CDP mouse events.

    Resolves source coordinates from element_id if available.
    Simulates: mousedown → slow mousemove → mouseup (with humanized Bézier path).
    Works for sliders, CAPTCHA puzzles, reorder lists, range inputs.
    """
    page = _get_page()
    try:
        # Resolve source from element registry if available
        if element_id:
            from agent_first_browse.perception import dom as dom_parser
            r = await dom_parser.resolve_element(page, element_id)
            if r.get("ok"):
                x, y = float(r["x"]), float(r["y"])

        if target_element_id:
            from agent_first_browse.perception import dom as dom_parser
            destination = await dom_parser.resolve_element(page, target_element_id)
            if not destination.get("ok"):
                return {"success": False, "error": "drag destination is no longer grounded"}
            target_x, target_y = float(destination["x"]), float(destination["y"])

        if x == 0 and y == 0:
            return {"success": False, "error": "drag_and_drop requires source coordinates or element_id"}
        if target_x == 0 and target_y == 0:
            return {"success": False, "error": "drag_and_drop requires target_x, target_y coordinates"}

        # Humanized move to source
        from agent_first_browse.browser.ghost_input import ghost_move_to
        await ghost_move_to(page, x, y)
        await asyncio.sleep(0.1)

        # Use a real Playwright mouse button sequence. Synthetic JS events do
        # not start native HTML5/pointer drags in many survey widgets.
        await page.mouse.down()
        await asyncio.sleep(0.15)

        # Humanized move to target (simulates dragging)
        # Move in steps for dragover events
        steps = 8
        for i in range(1, steps + 1):
            ix = x + (target_x - x) * i / steps
            iy = y + (target_y - y) * i / steps
            await page.mouse.move(ix, iy)
            await asyncio.sleep(0.03)

        # Dispatch dragover/mouseover at target
        await page.evaluate("""
        ([x, y]) => {
            const el = document.elementFromPoint(x, y);
            if (el) {
                el.dispatchEvent(new MouseEvent('mouseover', {
                    bubbles: true, clientX: x, clientY: y
                }));
                el.dispatchEvent(new MouseEvent('mousemove', {
                    bubbles: true, clientX: x, clientY: y
                }));
            }
        }
        """, [target_x, target_y])
        await asyncio.sleep(0.1)

        # Release the real held button at the destination.
        await page.mouse.up()

        logger.info("✅ drag_and_drop: (%.0f,%.0f) → (%.0f,%.0f)", x, y, target_x, target_y)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_drag_and_drop failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


async def mcp_find_drag_targets(source_text: str) -> dict:
    """Locate visually rendered drag source/drop zone when a11y omits them."""
    page = _get_page()
    try:
        result = await page.evaluate("""
        (wanted) => {
          const visible = (el) => {
            const r = el.getBoundingClientRect(), s = getComputedStyle(el);
            return r.width >= 8 && r.height >= 8 && r.bottom > 0 && r.right > 0
              && r.top < innerHeight && r.left < innerWidth
              && s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity || 1) > .05;
          };
          const nodes = [...document.querySelectorAll('body *')].filter(visible);
          const exact = nodes.filter(el => (el.innerText || el.textContent || '').trim() === wanted);
          const source = exact.sort((a,b) => {
            const score = el => {
              const s=getComputedStyle(el), c=String(el.className||'').toLowerCase();
              return (el.draggable?20:0) + (/drag|token|item/.test(c)?12:0)
                + (/grab|move/.test(s.cursor)?8:0) - el.children.length*2;
            }; return score(b)-score(a);
          })[0];
          if (!source) return {ok:false, reason:'source text not found'};
          const sr=source.getBoundingClientRect();
          const targets=nodes.filter(el => el!==source && !el.contains(source)).map(el => {
            const r=el.getBoundingClientRect(), s=getComputedStyle(el);
            const c=`${el.id||''} ${el.className||''} ${el.getAttribute('aria-label')||''}`.toLowerCase();
            const square=Math.abs(r.width-r.height) <= Math.max(12, Math.min(r.width,r.height)*.35);
            const semantic=/drop|target|square|box|bucket/.test(c);
            const bordered=parseFloat(s.borderTopWidth||0)>0 || s.boxShadow!=='none';
            const empty=(el.innerText||el.textContent||'').trim()==='';
            const distance=Math.hypot((r.x+r.width/2)-(sr.x+sr.width/2),(r.y+r.height/2)-(sr.y+sr.height/2));
            const score=(semantic?40:0)+(square?20:0)+(bordered?8:0)+(empty?8:0)
              +(r.width>=30&&r.width<=350&&r.height>=30&&r.height<=350?8:0)-distance/100;
            return {el,r,score};
          }).filter(x => x.score >= 15).sort((a,b)=>b.score-a.score);
          if (!targets.length) return {ok:false, reason:'drop zone not found'};
          const tr=targets[0].r;
          return {ok:true, source_x:sr.x+sr.width/2, source_y:sr.y+sr.height/2,
            target_x:tr.x+tr.width/2, target_y:tr.y+tr.height/2};
        }
        """, str(source_text).strip())
        return result if isinstance(result, dict) else {"ok": False, "reason": "invalid geometry result"}
    except Exception as exc:
        logger.warning("mcp_find_drag_targets failed: %s", exc)
        return {"ok": False, "reason": str(exc)[:160]}


async def mcp_upload_file(element_id: str | None, file_path: str) -> dict:
    """Upload a file to an <input type='file'> element.

    Uses Playwright's set_input_files() which bypasses the OS file picker.
    Resolves element via the __aid registry.
    """
    page = _get_page()
    if not element_id:
        return {"success": False, "error": "upload_file requires an element_id"}
    if not file_path:
        return {"success": False, "error": "upload_file requires a file_path"}
    try:
        import os
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path[:80]}"}

        # Find the actual input[type=file] element using __aid
        locator = page.locator(f'[__aid="{element_id}"]')
        count = await locator.count()
        if count == 0:
            # Fallback: try data-eid
            locator = page.locator(f'[data-eid="{element_id}"]')
            count = await locator.count()
        if count == 0:
            return {"success": False, "error": f"Element {element_id} not found in DOM"}

        await locator.first.set_input_files(file_path)
        logger.info("✅ upload_file: '%s' → %s", os.path.basename(file_path), element_id)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_upload_file failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


async def mcp_scroll_directional(direction: str = "down", amount: int = 500) -> dict:
    """Scroll the page in a specific direction by a given pixel amount.

    Supports: up, down, left, right. Uses page.mouse.wheel() for precision.
    For carousels, sidebars, infinite scrolls, and targeted navigation.
    """
    page = _get_page()
    direction = (direction or "down").lower().strip()
    amount = max(50, min(amount or 500, 5000))  # Clamp to [50, 5000]

    direction_map = {
        "down":  (0,  amount),
        "up":    (0, -amount),
        "right": (amount,  0),
        "left":  (-amount, 0),
    }

    if direction not in direction_map:
        return {"success": False, "error": f"Invalid direction '{direction}'. Use: up, down, left, right"}

    try:
        dx, dy = direction_map[direction]
        await page.mouse.wheel(dx, dy)
        await asyncio.sleep(0.3)  # Let scroll settle
        logger.info("✅ scroll_directional: %s by %dpx", direction, amount)
        return {"success": True, "error": ""}
    except Exception as e:
        logger.warning("mcp_scroll_directional failed: %s", e)
        return {"success": False, "error": str(e)[:160]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Snapshot (DOM Perception)
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_snapshot() -> dict:
    """Take an accessibility-tree snapshot of the current page.

    Delegates to: dom_parser.extract()
    Returns: {"elements": list, "markdown": str, "element_count": int, "selector_map": dict}
    """
    page = _get_page()
    try:
        qmee_popup = await mcp_close_qmee_svg_popup()
        if qmee_popup.get("closed"):
            logger.info("🪟 Closed Qmee SVG popup before DOM snapshot")
        from agent_first_browse.perception import dom as dom_parser
        dom_data = await dom_parser.extract(page, target_hint=None, timeout=5.0)
        elements_list = dom_data.get("elements", [])

        # Build selector map
        smap: dict[str, dict] = {}
        for el in elements_list:
            eid = el.get("id", el.get("ref", ""))
            if eid:
                smap[eid] = el

        # The ranked snapshot intentionally favors prompt compactness.  On a
        # survey question that can hide every option, perform one bounded
        # native audit before the worker is allowed to escalate to vision.
        sparse_recovery = {"status": "NOT_NEEDED", "controls": [], "count": 0}
        try:
            from agent_first_browse.survey.context import sparse_survey_dom
            if sparse_survey_dom(dom_data.get("page_text", ""), smap):
                sparse_recovery = await dom_parser.recover_sparse_controls(page)
                for recovered in sparse_recovery.get("controls", []):
                    eid = recovered.get("id")
                    if eid and eid not in smap:
                        elements_list.append(recovered)
                        smap[eid] = recovered
                if sparse_recovery.get("controls"):
                    recovery_lines = ["## Sparse DOM recovery"]
                    for el in sparse_recovery["controls"][:40]:
                        recovery_lines.append(
                            f"- **[{el.get('id')}]** {el.get('kind')}: {el.get('text') or '(unlabeled)'} "
                            f"→ ({el.get('x')},{el.get('y')})"
                            + (" [selected]" if el.get("selected") else "")
                        )
                    dom_data["markdown"] = (dom_data.get("markdown", "") + "\n\n" + "\n".join(recovery_lines)).strip()
                logger.info(
                    "🔎 Sparse DOM recovery status=%s controls=%d",
                    sparse_recovery.get("status"), sparse_recovery.get("count", 0),
                )
        except Exception as exc:
            logger.debug("Sparse DOM recovery skipped: %s", str(exc)[:160])

        # Update global selector map
        set_selector_map(smap)
        paidwork_ready = None
        try:
            from agent_first_browse.survey.context import paidwork_selection_ready
            paidwork_ready = paidwork_selection_ready(page.url, dom_data.get("page_text", ""), smap)
        except Exception:
            pass

        # One revision covers every representation returned from this pass.
        # Overwatch and workers can therefore reject proposals from an older
        # selector map instead of mixing sparse-recovery controls with the
        # original element count/handle registry.
        snapshot_revision = hashlib.sha256(json.dumps({
            "url": getattr(page, "url", ""),
            "elements": elements_list,
        }, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
        return {
            "elements": elements_list,
            "markdown": dom_data.get("markdown", ""),
            "page_text": dom_data.get("page_text", ""),
            "element_count": len(elements_list),
            "selector_map": smap,
            "snapshot_revision": snapshot_revision,
            "image_size": dom_data.get("image_size", {}),
            "sparse_dom_status": sparse_recovery.get("status", "NOT_NEEDED"),
            "sparse_dom_control_count": int(sparse_recovery.get("count", 0) or 0),
            "sparse_dom_reason": (
                "question_present_controls_missing"
                if sparse_recovery.get("status") != "NOT_NEEDED" else ""
            ),
            "paidwork_selection_ready": paidwork_ready,
        }
    except Exception as e:
        logger.warning("mcp_snapshot failed: %s", e)
        return {"elements": [], "markdown": "", "page_text": "", "element_count": 0,
                "selector_map": {}, "image_size": {}}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Screenshot (for on-demand vision consults)
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_screenshot(full_page: bool = False) -> dict:
    """Capture a base64-encoded viewport screenshot.

    Used ONLY when the agent escalates to vision (it cannot resolve the page from
    the a11y DOM alone). Viewport — not full-page — so the image aligns with the
    element-map coordinates the vision model reasons over.

    Returns: {"ok": bool, "base64": str, "error": str}
    """
    page = _get_page()
    try:
        # page.screenshot captures ONLY the browser viewport (the web content
        # area) — never the browser chrome or the OS desktop. So vision is
        # inherently confined to the page; we also pass the exact bounds on so the
        # model knows the coordinate frame (0,0 .. width,height).
        png = await asyncio.wait_for(
            page.screenshot(full_page=full_page, type="png"), timeout=10.0
        )
        vp = page.viewport_size or {"width": 0, "height": 0}
        if not vp.get("width") or not vp.get("height"):
            # When attached to an existing Chrome, Playwright often has no
            # configured viewport even though the page has a real viewport.
            # Vision coordinates must use the rendered browser dimensions.
            try:
                live_vp = await page.evaluate("""
                    () => ({
                        width: Math.round(window.innerWidth || document.documentElement.clientWidth || 0),
                        height: Math.round(window.innerHeight || document.documentElement.clientHeight || 0),
                    })
                """)
                vp = live_vp or vp
            except Exception:
                pass
        return {"ok": True, "base64": base64.b64encode(png).decode("utf-8"),
                "width": int(vp.get("width", 0)), "height": int(vp.get("height", 0)),
                "error": ""}
    except Exception as e:
        logger.warning("mcp_screenshot failed: %s", e)
        return {"ok": False, "base64": "", "width": 0, "height": 0, "error": str(e)[:200]}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Wait
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_wait(ms: int = 800) -> dict:
    """Wait for a specified number of milliseconds.

    Returns: {"success": bool, "waited_ms": int}
    """
    page = _get_page()
    try:
        await page.wait_for_timeout(ms)
        return {"success": True, "waited_ms": ms}
    except Exception as e:
        return {"success": False, "waited_ms": 0}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Login State Detection
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_detect_login() -> dict:
    """Detect login state of the current page.

    Returns: {"logged_in": bool, "has_login_form": bool}
    """
    page = _get_page()
    try:
        login_state = await asyncio.wait_for(page.evaluate("""
        () => {
            const hasProfile = !!document.querySelector(
                '[aria-label*="profile" i], [aria-label*="account" i], '
                + 'img[alt*="avatar" i], img[alt*="profile" i], '
                + '[data-testid*="profile" i], [data-testid*="user" i], '
                + '.user-menu, .profile-menu, #user-nav'
            );
            let hasLoginForm = !!document.querySelector(
                'input[type="password"], '
                + 'form[action*="login" i], form[action*="signin" i]'
            );
            if (!hasLoginForm) {
                const buttons = document.querySelectorAll('button, a[role="button"], input[type="submit"]');
                for (const btn of buttons) {
                    const txt = (btn.textContent || '').trim().toLowerCase();
                    if (['sign in', 'log in', 'login', 'signin'].includes(txt)) {
                        hasLoginForm = true;
                        break;
                    }
                }
            }
            return { hasProfile, hasLoginForm };
        }
        """), timeout=5.0)

        return {
            "logged_in": login_state.get("hasProfile", False),
            "has_login_form": login_state.get("hasLoginForm", False),
        }
    except Exception:
        return {"logged_in": False, "has_login_form": False}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool: Grounding Validation
# ═══════════════════════════════════════════════════════════════════════════════

async def mcp_ground_action(
    element_id: str | None,
    x: float | None,
    y: float | None,
    selector_map: dict | None = None,
    elements_list: list | None = None,
) -> dict:
    """Validate and ground an action's target coordinates.

    Delegates to: advanced_agent._ground_or_reject() logic
    Returns: {"grounded": bool, "x": float, "y": float, "element": dict|None, "reason": str}
    """
    page = _get_page()
    smap = selector_map or _SELECTOR_MAP

    # Layer 0 (current): Live registry resolution — the EXACT node the LLM chose, with
    # fresh coordinates. Supersedes the "snap to nearest within 60px" heuristic
    # that could land on the wrong neighbour on dense pages.
    if element_id is not None:
        from agent_first_browse.perception import dom as dom_parser
        r = await dom_parser.resolve_element(page, element_id)
        if r.get("ok"):
            return {"grounded": True, "x": float(r["x"]), "y": float(r["y"]),
                    "element": {"ref": element_id, "name": r.get("text", "")},
                    "reason": "registry resolved (fresh coords)"}

    # Layer 1: Element ID resolution
    if element_id is not None:
        el = smap.get(element_id)
        if el is None:
            return {"grounded": False, "x": 0, "y": 0, "element": None,
                    "reason": f"element_id '{element_id}' not in current snapshot"}
        resolved_x = float(el.get("x", 0))
        resolved_y = float(el.get("y", 0))
        return {"grounded": True, "x": resolved_x, "y": resolved_y,
                "element": el, "reason": "element_id resolved"}

    if x is None or y is None:
        return {"grounded": False, "x": 0, "y": 0, "element": None,
                "reason": "no element_id or coordinates"}

    # Layer 2: elementFromPoint hit-test
    try:
        hit = await page.evaluate(
            "([x,y])=>{const e=document.elementFromPoint(x,y);"
            "if(!e)return null;"
            "const tag=e.tagName.toUpperCase();"
            "const interactive=['A','BUTTON','INPUT','SELECT','TEXTAREA']"
            ".includes(tag)||e.getAttribute('role')==='button'"
            "||e.getAttribute('contenteditable')==='true';"
            "return{tag,interactive,text:(e.textContent||'').trim().slice(0,40)};}",
            [x, y],
        )
        if hit and hit.get("interactive"):
            return {"grounded": True, "x": x, "y": y,
                    "element": {"ref": "hit-test", "name": hit.get("text", "")[:30]},
                    "reason": "hit-test grounded"}
    except Exception:
        pass

    # Layer 3: Nearest-element snap
    elems = elements_list or []
    if not elems:
        return {"grounded": False, "x": x, "y": y, "element": None,
                "reason": "no elements in snapshot"}

    best_el = None
    best_dist = float("inf")
    for el in elems:
        ex = el.get("x", 0)
        ey = el.get("y", 0)
        if ex == 0 and ey == 0:
            continue
        dist = ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_el = el

    threshold = 60.0
    if best_el is None or best_dist > threshold:
        return {"grounded": False, "x": x, "y": y, "element": None,
                "reason": f"nearest element is {best_dist:.0f}px away (>{threshold:.0f}px)"}

    return {"grounded": True, "x": float(best_el.get("x", x)),
            "y": float(best_el.get("y", y)), "element": best_el,
            "reason": f"snapped to nearest ({best_dist:.0f}px)"}
