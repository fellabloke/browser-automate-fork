"""CDP-Native Click Engine — Multi-strategy click injection for anti-bot-protected sites.

Provides trusted click methods that bypass Amazon/Shopify/Reddit JS overlay traps
where Playwright's synthetic mouse.click() is silently dropped.

Click Strategy Waterfall:
  1. CDP Input.dispatchMouseEvent (trusted OS-level mouse events)
  2. CDP DOM.focus + JS element.click() (programmatic click via DOM)
  3. Playwright page.mouse.click() (legacy fallback)

Each strategy includes post-click verification to confirm the page responded.

Architecture:
  - overlay_detector.py runs BEFORE this module (neutralizes pointer-events traps)
  - ghost_input.py handles the Bézier mouse movement path
  - This module dispatches the actual click event at the destination

Why Playwright's mouse.click() fails on Amazon:
  Amazon's anti-bot JS listens for the `isTrusted` flag on MouseEvents.
  Playwright dispatches events via the Blink event queue, but certain sites
  filter events that don't originate from the browser's input pipeline.
  CDP's Input.dispatchMouseEvent goes through the browser's native input
  pipeline and produces events with isTrusted=true.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import Page

try:
    from app.logger import get_logger
    logger = get_logger("cdp_click")
except ImportError:
    import logging
    logger = logging.getLogger("cdp_click")


# ═══════════════════════════════════════════════════════════════════════════════
#  Result Types
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClickResult:
    """Result of a click attempt."""
    success: bool
    strategy: str           # Which strategy produced the click
    navigation: bool        # Did the page navigate after the click?
    dom_changed: bool       # Did the DOM structure change?
    attempts: int           # How many strategies were tried
    error: str = ""         # Error message if failed
    pre_url: str = ""       # URL before click
    post_url: str = ""      # URL after click
    verified: bool = True    # Was an observable post-click effect confirmed?
    dispatched: bool = False # Did at least one click event reach the page?
    no_op: bool = False      # Was dispatch safely suppressed as redundant/harmful?


def _navigation_identity(url: str) -> tuple[str, str, str, str]:
    """Navigation identity that deliberately ignores fragment-only churn."""
    try:
        parsed = urlsplit(str(url or ""))
        return parsed.scheme, parsed.netloc, parsed.path, parsed.query
    except ValueError:
        return "", "", str(url or ""), ""


def _is_validation_rejection(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return any(marker in normalized for marker in (
        "problem", "invalid", "required", "please enter", "please select",
        "not accepted", "please correct", "must be", "validation error",
    ))


# ═══════════════════════════════════════════════════════════════════════════════
#  CDP Session Helper
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_cdp_session(page: Page):
    """Create a CDP session from a Playwright page.
    
    Returns None on failure instead of raising — the caller must handle it.
    """
    try:
        return await page.context.new_cdp_session(page)
    except Exception as e:
        logger.warning("CDP session creation failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre/Post Click State Capture
# ═══════════════════════════════════════════════════════════════════════════════

async def _capture_page_fingerprint(page: Page, timeout: float = 2.0) -> dict:
    """Capture a lightweight fingerprint of the page for change detection.
    
    Returns:
        {
            'url': str,
            'title': str,
            'element_count': int,
            'structural_hash': str,  # MD5 of interactive element structure
        }
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        () => {
            const interactives = document.querySelectorAll(
                'a, button, input, textarea, select, [role="button"], [role="link"], [role="menuitem"]'
            );
            const parts = [];
            interactives.forEach((el, i) => {
                if (i < 100) {  // Cap at 100 elements for performance
                    const rect = el.getBoundingClientRect();
                    parts.push(
                        el.tagName + '|' +
                        (el.textContent || '').trim().slice(0, 20) + '|' +
                        Math.round(rect.x) + ',' + Math.round(rect.y)
                    );
                }
            });
            const validationSelectors = [
                '[role="alert"]', '[aria-invalid="true"]',
                '.error', '.field-error', '.validation-error', '.invalid-feedback',
                '[class*="error"]', '[class*="invalid"]', '[data-error]',
                '.sg-error-message', '.sg-question-error'
            ];
            const validationText = [];
            document.querySelectorAll(validationSelectors.join(',')).forEach((el) => {
                const style = getComputedStyle(el);
                const text = (el.textContent || el.validationMessage || '').trim();
                if (text && style.display !== 'none' && style.visibility !== 'hidden') {
                    validationText.push(text.slice(0, 180));
                }
            });
            /* Some survey engines render validation in an unclassed text node.
               Inspect only short, leaf-ish visible nodes to avoid interpreting
               neutral instructions such as "fields marked * are required" as
               a live rejection. */
            const validationPattern = /(?:there (?:was|were) (?:an? )?(?:error|problem)|problems? with (?:some )?(?:data|answers?)|(?:this |the )?(?:answer|field|question|selection|response) (?:is )?required|required fields?|please (?:answer|choose|correct|enter|provide|select).{0,45}(?:before (?:continuing|proceeding)|to continue|required|highlighted|missing)|invalid (?:answer|entry|response|selection|value)|validation error)/i;
            for (const el of document.querySelectorAll('div,span,p,li,label,small')) {
                if (el.children.length > 2) continue;
                const text = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
                if (!text || text.length > 240 || !validationPattern.test(text)) continue;
                const style = getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                if (rect.width > 1 && rect.height > 1 && style.display !== 'none'
                    && style.visibility !== 'hidden' && parseFloat(style.opacity || '1') > 0.01) {
                    validationText.push(text.slice(0, 180));
                }
                if (validationText.length >= 6) break;
            }
            return {
                url: location.href,
                title: document.title,
                elementCount: interactives.length,
                structure: parts.join(';;'),
                validationError: validationText.join(' | ').slice(0, 500)
            };
        }
        """), timeout=timeout)
        
        structure = result.get("structure", "")
        structural_hash = hashlib.md5(structure.encode()).hexdigest()[:12]
        
        return {
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "element_count": result.get("elementCount", 0),
            "structural_hash": structural_hash,
            "validation_error": result.get("validationError", ""),
        }
    except Exception as e:
        logger.debug("Page fingerprint capture failed: %s", e)
        return {
            "url": "",
            "title": "",
            "element_count": 0,
            "structural_hash": "",
            "validation_error": "",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: CDP Input.dispatchMouseEvent (Trusted Native Events)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_cdp_mouse_event(
    page: Page,
    x: float,
    y: float,
    button: str = "left",
) -> bool:
    """Dispatch a full mouse click via CDP Input.dispatchMouseEvent.
    
    This goes through the browser's native input pipeline and produces
    MouseEvents with isTrusted=true, which is the key property that
    anti-bot scripts check.
    
    Sequence: mouseMoved → mousePressed → mouseReleased
    
    Each event has realistic timing jitter and proper modifier flags.
    """
    cdp = await _get_cdp_session(page)
    if not cdp:
        return False
    
    try:
        # Map button name to CDP button code
        button_map = {"left": "left", "right": "right", "middle": "middle"}
        cdp_button = button_map.get(button, "left")
        click_count = 1
        
        # Realistic timestamp base (microseconds since epoch)
        ts = time.time()
        
        # Step 1: Mouse moved to position (generates mouseover/mouseenter events)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": int(x),
            "y": int(y),
            "button": "none",
            "timestamp": ts,
        })
        await asyncio.sleep(random.uniform(0.02, 0.06))
        
        # Step 2: Mouse pressed (generates mousedown event)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": int(x),
            "y": int(y),
            "button": cdp_button,
            "clickCount": click_count,
            "timestamp": ts + random.uniform(0.03, 0.08),
        })
        
        # Realistic press-to-release delay (humans hold for 50-150ms)
        await asyncio.sleep(random.uniform(0.05, 0.15))
        
        # Step 3: Mouse released (generates mouseup + click events)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": int(x),
            "y": int(y),
            "button": cdp_button,
            "clickCount": click_count,
            "timestamp": ts + random.uniform(0.10, 0.20),
        })
        
        logger.debug("CDP mouse event click dispatched at (%d, %d)", x, y)
        return True
        
    except Exception as e:
        logger.warning("CDP mouse event failed at (%d, %d): %s", x, y, e)
        return False
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: CDP DOM.focus + JS element.click()
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_js_click(
    page: Page,
    x: float,
    y: float,
) -> bool:
    """Find the element at (x, y) and call its .click() method directly.
    
    This bypasses the mouse event pipeline entirely and triggers the
    element's click handler programmatically. Less realistic but more
    reliable when event listeners are filtered.
    
    Also dispatches synthetic pointer/mouse down/up events for frameworks
    that listen on those instead of the click handler. Exactly ONE click event
    is emitted: ``el.click()`` supplies it. Dispatching a separate synthetic
    ``click`` and then calling ``el.click()`` invokes toggle handlers twice.
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        (params) => {
            const { x, y } = params;
            const el = document.elementFromPoint(x, y);
            if (!el) return { clicked: false, reason: 'no_element_at_point' };

            // Survey widgets frequently hide the native radio input and put
            // the real handler on its visible label/card. A coordinate from
            // vision must activate that visual control, not call .click() on
            // an inert child span or the hidden input itself.
            const target = el.closest && (
                el.closest('label,[role="radio"],[role="checkbox"],[role="option"]')
                || el.closest('button,[role="button"]')
            ) || el;
            
            const tag = el.tagName.toLowerCase();
            const text = (el.textContent || '').trim().slice(0, 40);
            
            try {
                // Step 1: Focus the element (if focusable)
                if (typeof target.focus === 'function') {
                    target.focus();
                }
                
                // Step 2: Dispatch the press/release sequence. Do NOT dispatch
                // a synthetic click here: el.click() below emits the one and
                // only click event for this strategy.
                const eventInit = {
                    bubbles: true,
                    cancelable: true,
                    view: window,
                    clientX: x,
                    clientY: y,
                    screenX: x,
                    screenY: y,
                    button: 0,
                    buttons: 1,
                };
                
                target.dispatchEvent(new PointerEvent('pointerdown', eventInit));
                target.dispatchEvent(new MouseEvent('mousedown', eventInit));
                target.dispatchEvent(new PointerEvent('pointerup', eventInit));
                target.dispatchEvent(new MouseEvent('mouseup', eventInit));
                
                // Step 3: Emit exactly one click.
                // Keep the original single-click invariant for the normal
                // element path; use the promoted visual control only when the
                // coordinate landed on a nested/inert child.
                if (target === el) el.click();
                else target.click();
                
                return {
                    clicked: true,
                    tag: target.tagName ? target.tagName.toLowerCase() : tag,
                    text: (target.textContent || text || '').trim().slice(0, 40),
                    reason: 'js_click_dispatched'
                };
            } catch (e) {
                return { clicked: false, reason: 'click_error: ' + e.message };
            }
        }
        """, {"x": x, "y": y}), timeout=3.0)
        
        if result and result.get("clicked"):
            logger.debug(
                "JS click dispatched at (%d, %d) on <%s> '%s'",
                x, y, result.get("tag", "?"), result.get("text", "")[:30],
            )
            return True
        else:
            reason = result.get("reason", "unknown") if result else "eval_returned_none"
            logger.warning("JS click failed at (%d, %d): %s", x, y, reason)
            return False
            
    except Exception as e:
        logger.warning("JS click failed at (%d, %d): %s", x, y, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Playwright page.mouse.click() (Legacy Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_playwright_click(
    page: Page,
    x: float,
    y: float,
) -> bool:
    """Standard Playwright mouse click — the existing fallback.
    
    This is what ghost_input.ghost_click() ultimately calls.
    Kept as the final fallback for sites that don't filter synthetic events.
    """
    try:
        await page.mouse.click(x, y)
        logger.debug("Playwright click dispatched at (%d, %d)", x, y)
        return True
    except Exception as e:
        logger.warning("Playwright click failed at (%d, %d): %s", x, y, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 4: Navigate via href (for <a> elements)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_direct_navigate(
    page: Page,
    x: float,
    y: float,
) -> bool:
    """If the element at (x, y) is an <a> tag, navigate directly to its href.
    
    This completely bypasses click event handling and is the most reliable
    way to follow links on heavily protected sites.
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        (params) => {
            const { x, y } = params;
            let el = document.elementFromPoint(x, y);
            if (!el) return { found: false };
            
            // Walk up to find the nearest <a> ancestor
            let current = el;
            for (let i = 0; i < 5 && current; i++) {
                if (current.tagName && current.tagName.toLowerCase() === 'a' && current.href) {
                    return {
                        found: true,
                        href: current.href,
                        text: (current.textContent || '').trim().slice(0, 40),
                        target: current.target || '_self',
                    };
                }
                current = current.parentElement;
            }
            return { found: false };
        }
        """, {"x": x, "y": y}), timeout=2.0)
        
        if result and result.get("found") and result.get("href"):
            href = result["href"]
            target = result.get("target", "_self")
            
            # Don't navigate to javascript: or # URLs
            if href.startswith("javascript:") or href == "#":
                logger.debug("Direct navigate skipped: href=%s", href[:60])
                return False
            
            logger.info(
                "🔗 Direct navigate to href: %s (text: '%s')",
                href[:80], result.get("text", "")[:30],
            )
            
            # Navigate in the current tab
            if target == "_blank":
                # For _blank targets, open in same tab to keep control
                await page.goto(href, wait_until="domcontentloaded", timeout=15000)
            else:
                await page.goto(href, wait_until="domcontentloaded", timeout=15000)
            
            return True
        
        return False
        
    except Exception as e:
        logger.warning("Direct navigate failed at (%d, %d): %s", x, y, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  V32: Element State Capture (Visual Truth Override)
# ═══════════════════════════════════════════════════════════════════════════════

async def _capture_element_state(
    page: Page,
    x: float,
    y: float,
    timeout: float = 2.0,
    element_id: str | None = None,
) -> dict:
    """Capture the interactive state of the element at (x, y).

    Prefer the exact ``window.__aid`` node selected by the worker, then capture
    it, a stateful descendant, and nearby wrappers. Custom React/Vue survey
    choices commonly put selection state on a wrapper class or ``data-state``
    rather than a native input, so coordinate-only/native-attribute capture is
    insufficient.

    This is the foundation of the Visual Truth Override — it lets us detect
    that a click WORKED even when the structural fingerprint sees no change.
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        ({x, y, elementId}) => {
            const registered = elementId && window.__aid
                ? window.__aid[elementId] : null;
            const exact = !!(registered && registered.isConnected);
            const el = exact ? registered : document.elementFromPoint(x, y);
            if (!el) return {found: false};

            const STATEFUL = [
                'input', 'select', 'textarea', 'option',
                '[role="checkbox"]', '[role="radio"]', '[role="switch"]',
                '[role="option"]', '[role="tab"]',
                '[aria-checked]', '[aria-selected]', '[aria-pressed]',
                '[data-state]', '[data-selected]', '[data-checked]'
            ].join(',');

            // Labels/wrappers often contain the native input rather than being
            // its ancestor. Search both directions and honour HTMLLabel.control.
            let control = null;
            if (el.matches && el.matches(STATEFUL)) control = el;
            if (!control && el.querySelector) control = el.querySelector(STATEFUL);
            if (!control && el.closest) control = el.closest(STATEFUL);
            const label = el.closest ? el.closest('label') : null;
            if (!control && label) control = label.control || label.querySelector(STATEFUL);
            control = control || el;

            const nodes = [];
            const seen = new Set();
            const add = (key, node) => {
                if (!node || seen.has(node)) return;
                seen.add(node);
                const cs = window.getComputedStyle(node);
                nodes.push({
                    key,
                    tag: node.tagName || '',
                    checked: !!node.checked,
                    selected: !!node.selected,
                    disabled: !!node.disabled,
                    value: (node.value || '').slice(0, 100),
                    ariaChecked: node.getAttribute('aria-checked') || '',
                    ariaSelected: node.getAttribute('aria-selected') || '',
                    ariaExpanded: node.getAttribute('aria-expanded') || '',
                    ariaPressed: node.getAttribute('aria-pressed') || '',
                    dataState: node.getAttribute('data-state') || '',
                    dataSelected: node.getAttribute('data-selected') || '',
                    dataChecked: node.getAttribute('data-checked') || '',
                    classList: Array.from(node.classList || []).join(' ').slice(0, 240),
                    visualStyle: [cs.backgroundColor, cs.borderColor,
                                  cs.boxShadow, cs.outlineColor].join('|').slice(0, 300),
                });
            };
            add('target', el);
            add('control', control);
            let parent = el.parentElement;
            for (let i = 0; i < 3 && parent; i++, parent = parent.parentElement) {
                add('parent' + i, parent);
            }

            const primary = nodes.find(n => n.key === 'control') || nodes[0];
            const link = el.closest ? el.closest('a[href], [role="link"]') : null;
            return {
                found: true,
                exactTarget: exact,
                interactionKind: link ? 'link' : 'control',
                tag: primary.tag,
                type: (control.type || '').toLowerCase(),
                checked: primary.checked,
                selected: primary.selected,
                value: primary.value,
                ariaChecked: primary.ariaChecked,
                ariaSelected: primary.ariaSelected,
                ariaExpanded: primary.ariaExpanded,
                ariaPressed: primary.ariaPressed,
                dataState: primary.dataState,
                dataSelected: primary.dataSelected,
                dataChecked: primary.dataChecked,
                classList: primary.classList,
                signals: nodes,
            };
        }
        """, {"x": x, "y": y, "elementId": element_id}), timeout=timeout)
        return result
    except Exception as e:
        logger.debug("Element state capture failed: %s", e)
        return {"found": False}


def _element_state_changed(pre: dict, post: dict) -> tuple[bool, list[str]]:
    """Compare pre/post element state. Returns (changed, list_of_changes).

    Checks native state plus custom-control data attributes, wrapper classes,
    and stable visual style. The pointer is already resting on the target when
    pre-state is captured, so hover styling is present in both snapshots and
    does not masquerade as a click effect.
    """
    if not pre.get("found") or not post.get("found"):
        return False, []

    changes = []
    # Computed colours can fluctuate while a CSS transition is settling (and
    # Chromium may serialize the same colour through rgb/oklab differently).
    # A visual-style delta is useful supporting diagnostics, but is not proof
    # that a choice changed.  Repeated clicks on an already-selected radio used
    # to be reported as success based solely on this cosmetic noise.
    state_attrs = (
        "checked", "selected", "disabled", "value", "ariaChecked",
        "ariaSelected", "ariaExpanded", "ariaPressed", "dataState",
        "dataSelected", "dataChecked", "classList",
    )
    for attr in state_attrs:
        old = pre.get(attr)
        new = post.get(attr)
        if old is not None and new is not None and old != new:
            changes.append(f"{attr}: {old!r} → {new!r}")

    pre_signals = {s.get("key"): s for s in pre.get("signals", [])}
    post_signals = {s.get("key"): s for s in post.get("signals", [])}
    for key in pre_signals.keys() & post_signals.keys():
        before = pre_signals[key]
        after = post_signals[key]
        for attr in state_attrs:
            old = before.get(attr)
            new = after.get(attr)
            if old is not None and new is not None and old != new:
                change = f"{key}.{attr}: {old!r} → {new!r}"
                if change not in changes:
                    changes.append(change)

    # Include visual deltas only when a durable state signal also changed.
    # This preserves useful logs without allowing animation/hover noise to
    # trigger the Visual Truth Override by itself.
    if changes:
        for key in pre_signals.keys() & post_signals.keys():
            before = pre_signals[key]
            after = post_signals[key]
            old = before.get("visualStyle")
            new = after.get("visualStyle")
            if old is not None and new is not None and old != new:
                changes.append(f"{key}.visualStyle: {old!r} → {new!r}")

    return len(changes) > 0, changes


def _element_state_is_selected(state: dict) -> bool:
    """Recognize selected state across native and custom survey controls."""
    if not state.get("found"):
        return False
    candidates = [state, *(state.get("signals") or [])]
    for candidate in candidates:
        if candidate.get("checked") is True or candidate.get("selected") is True:
            return True
        for key in (
            "ariaChecked", "ariaSelected", "ariaPressed",
            "dataSelected", "dataChecked",
        ):
            if str(candidate.get(key) or "").strip().lower() == "true":
                return True
        if str(candidate.get("dataState") or "").strip().lower() in {
            "checked", "selected", "on",
        }:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point: Resilient Click with Waterfall
# ═══════════════════════════════════════════════════════════════════════════════

async def resilient_click(
    page: Page,
    x: float,
    y: float,
    button: str = "left",
    max_retries: int = 4,
    settle_ms: int = 800,
    element_id: str | None = None,
    prevent_deselect: bool = False,
    replay_safe: bool = False,
) -> ClickResult:
    """Click at (x, y) using a multi-strategy waterfall with post-click verification.
    
    Strategy waterfall (tries each until the page responds):
      1. CDP Input.dispatchMouseEvent — trusted native events (best for anti-bot sites)
      2. JS element.click() with full event sequence — programmatic fallback
      3. Direct href navigation — for <a> tags that ignore click events
      4. Playwright mouse.click() — legacy fallback
    
    Post-click verification:
      After each strategy, waits settle_ms and checks if:
      - The URL changed (navigation)
      - The DOM structure changed (modal/dropdown/state change)
      If neither changed, the click was likely silently dropped → try next strategy.
    
    Args:
        page: Playwright Page instance
        x: X coordinate to click
        y: Y coordinate to click  
        button: Mouse button ('left', 'right', 'middle')
        max_retries: Maximum number of strategies to try
        settle_ms: Milliseconds to wait after click for page to respond
        element_id: Stable DOM registry id. For an exact non-link target, a
            dispatched click is never replayed automatically (at-most-once).
    
    Returns:
        ClickResult with success status, strategy used, and diagnostics
    """
    # Capture pre-click state
    pre_fp = await _capture_page_fingerprint(page)
    pre_url = pre_fp.get("url", "")

    # V32: Capture element state BEFORE clicking (for toggleable controls)
    pre_element_state = await _capture_element_state(
        page, x, y, element_id=element_id
    )

    # Survey answer clicks mean "make this choice selected", not "toggle its
    # state blindly". The live resolver sees wrapper/control attributes that a
    # compressed DOM snapshot can miss. Suppress the click before it can undo a
    # completed checkbox/radio answer.
    if prevent_deselect and _element_state_is_selected(pre_element_state):
        logger.info(
            "Selected-state preflight: suppressing click that would deselect %s",
            element_id or f"({int(x)},{int(y)})",
        )
        return ClickResult(
            success=True,
            strategy="selected_state_no_op",
            navigation=False,
            dom_changed=False,
            attempts=0,
            pre_url=pre_url,
            post_url=pre_url,
            verified=False,
            dispatched=False,
            no_op=True,
        )

    # Replaying an unverified button/toggle click can undo a selection, submit
    # twice, or skip a survey question. Links are replay-safe because success is
    # unambiguously observable as navigation and direct href navigation remains
    # a useful fallback. Legacy coordinate-only callers retain the waterfall.
    at_most_once = bool(
        element_id
        and pre_element_state.get("exactTarget")
        and pre_element_state.get("interactionKind") != "link"
        and not replay_safe
    )
    
    native = ("cdp_mouse_event", lambda: _strategy_cdp_mouse_event(page, x, y, button))
    javascript = ("js_click", lambda: _strategy_js_click(page, x, y))
    direct = ("direct_navigate", lambda: _strategy_direct_navigate(page, x, y))
    playwright = ("playwright_click", lambda: _strategy_playwright_click(page, x, y))
    # For links, try the non-replaying href fallback before synthetic clicks.
    strategies = ([native, direct, javascript, playwright]
                  if pre_element_state.get("interactionKind") == "link"
                  else [native, javascript, direct, playwright])
    
    for attempt, (strategy_name, strategy_fn) in enumerate(strategies[:max_retries]):
        logger.info(
            "Click attempt %d/%d via %s at (%d, %d)",
            attempt + 1, max_retries, strategy_name, int(x), int(y),
        )
        
        try:
            # Execute the click strategy
            ok = await asyncio.wait_for(strategy_fn(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("%s timed out at (%d, %d)", strategy_name, int(x), int(y))
            continue
        except Exception as e:
            logger.warning("%s error at (%d, %d): %s", strategy_name, int(x), int(y), e)
            continue
        
        if not ok:
            logger.debug("%s returned False — trying next strategy", strategy_name)
            continue
        
        # Most control changes occur in the same task/microtask. Sample quickly
        # first; only replay-safe navigation controls pay the remaining legacy
        # settle budget before another strategy is attempted.
        fast_settle_ms = max(40, min(
            settle_ms,
            int(os.getenv("CLICK_FAST_SETTLE_MS", "120")),
        ))
        await asyncio.sleep(fast_settle_ms / 1000.0)
        
        # Check for page load state
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass  # Page might not need full load for the click to have effect
        
        # Capture post-click state
        post_fp = await _capture_page_fingerprint(page)
        post_url = post_fp.get("url", "")
        
        # Determine if the click had any effect
        navigated = bool(
            post_url
            and _navigation_identity(post_url) != _navigation_identity(pre_url)
        )
        dom_changed = (
            post_fp.get("structural_hash") != pre_fp.get("structural_hash")
            and post_fp.get("structural_hash") != ""
        )
        element_delta = abs(
            post_fp.get("element_count", 0) - pre_fp.get("element_count", 0)
        )
        
        validation_rejected = (
            _is_validation_rejection(str(post_fp.get("validation_error") or ""))
            and not navigated
        )
        if validation_rejected:
            logger.warning(
                "Click produced/retained a visible validation error; DOM/hash-fragment churn is not progress: %s",
                str(post_fp.get("validation_error"))[:140],
            )
        click_had_effect = (
            navigated or dom_changed or element_delta >= 2
        ) and not validation_rejected
        
        if click_had_effect:
            effect_desc = []
            if navigated:
                effect_desc.append(f"navigated to {post_url[:60]}")
            if dom_changed:
                effect_desc.append("DOM structure changed")
            if element_delta >= 2:
                effect_desc.append(f"element count changed by {element_delta}")
            
            logger.info(
                "✅ CLICK SUCCESS via %s at (%d, %d): %s",
                strategy_name, int(x), int(y), "; ".join(effect_desc),
            )
            return ClickResult(
                success=True,
                strategy=strategy_name,
                navigation=navigated,
                dom_changed=dom_changed,
                attempts=attempt + 1,
                pre_url=pre_url,
                post_url=post_url,
                dispatched=True,
            )
        
        # ── V32 VISUAL TRUTH OVERRIDE ──────────────────────────────────────
        # For toggleable elements (radio buttons, checkboxes, custom toggles),
        # clicking changes ONLY the element's state attributes (checked,
        # selected, value, aria-*) — NOT the DOM structure, URL, or element
        # count. The fingerprint hash is blind to these changes.
        #
        # CRITICAL: We check here INSIDE the loop, BEFORE trying the next
        # strategy. If we let strategy 2 fire on a checkbox that strategy 1
        # already toggled, it will UNDO the toggle (double-click problem).
        # ──────────────────────────────────────────────────────────────────
        if pre_element_state.get("found"):
            post_element_state = await _capture_element_state(
                page, x, y, element_id=element_id
            )
            changed, change_list = _element_state_changed(
                pre_element_state, post_element_state
            )
            if changed:
                logger.info(
                    "✅ VISUAL TRUTH OVERRIDE via %s at (%d, %d): "
                    "click appeared to fail but element state changed: %s",
                    strategy_name, int(x), int(y),
                    "; ".join(change_list),
                )
                return ClickResult(
                    success=True,
                    strategy=f"{strategy_name}+state_verified",
                    navigation=False,
                    dom_changed=False,
                    attempts=attempt + 1,
                    pre_url=pre_url,
                    post_url=pre_url,
                    dispatched=True,
                )

        remaining_settle_ms = max(0, settle_ms - fast_settle_ms)
        if remaining_settle_ms:
            await asyncio.sleep(remaining_settle_ms / 1000.0)
            delayed_fp = await _capture_page_fingerprint(page)
            delayed_url = delayed_fp.get("url", "")
            delayed_navigated = bool(
                delayed_url
                and _navigation_identity(delayed_url) != _navigation_identity(pre_url)
            )
            delayed_validation = _is_validation_rejection(
                str(delayed_fp.get("validation_error") or "")
            ) and not delayed_navigated
            delayed_changed = bool(
                delayed_fp.get("structural_hash")
                and delayed_fp.get("structural_hash") != pre_fp.get("structural_hash")
            )
            if (delayed_navigated or delayed_changed) and not delayed_validation:
                return ClickResult(
                    success=True,
                    strategy=f"{strategy_name}+delayed_verified",
                    navigation=delayed_navigated,
                    dom_changed=delayed_changed,
                    attempts=attempt + 1,
                    pre_url=pre_url,
                    post_url=delayed_url,
                    dispatched=True,
                )
            if at_most_once and pre_element_state.get("found"):
                delayed_element_state = await _capture_element_state(
                    page, x, y, element_id=element_id
                )
                changed, change_list = _element_state_changed(
                    pre_element_state, delayed_element_state
                )
                if changed:
                    logger.info(
                        "✅ Delayed element-state verification via %s: %s",
                        strategy_name, "; ".join(change_list),
                    )
                    return ClickResult(
                        success=True,
                        strategy=f"{strategy_name}+delayed_state_verified",
                        navigation=False,
                        dom_changed=False,
                        attempts=attempt + 1,
                        pre_url=pre_url,
                        post_url=delayed_url or pre_url,
                        dispatched=True,
                    )

        if at_most_once:
            logger.warning(
                "⚠️ %s dispatched once at (%d, %d) with no verifiable "
                "response — replay suppressed for non-link element %s",
                strategy_name, int(x), int(y), element_id,
            )
            return ClickResult(
                success=True,
                strategy=f"{strategy_name}+single_dispatch_unverified",
                navigation=False,
                dom_changed=False,
                attempts=attempt + 1,
                error="Click dispatched once; automatic replay suppressed",
                pre_url=pre_url,
                post_url=pre_url,
                verified=False,
                dispatched=True,
            )
        
        # Click dispatched but no visible effect — try next strategy
        logger.warning(
            "⚠️ %s dispatched at (%d, %d) but page did not respond — trying next",
            strategy_name, int(x), int(y),
        )
    
    # All strategies exhausted
    logger.error(
        "❌ CLICK FAILED: all %d strategies exhausted at (%d, %d)",
        min(max_retries, len(strategies)), int(x), int(y),
    )
    return ClickResult(
        success=False,
        strategy="none",
        navigation=False,
        dom_changed=False,
        attempts=min(max_retries, len(strategies)),
        error="All click strategies exhausted — page did not respond",
        pre_url=pre_url,
        post_url=pre_url,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: CDP-Only Click (for overlay bypass integration)
# ═══════════════════════════════════════════════════════════════════════════════

async def cdp_click(
    page: Page,
    x: float,
    y: float,
    button: str = "left",
) -> bool:
    """Simple CDP click without verification. Used by overlay_detector.
    
    Returns True if the CDP event was dispatched (not whether it had effect).
    """
    return await _strategy_cdp_mouse_event(page, x, y, button)
