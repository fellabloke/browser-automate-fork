"""CDP-Native Input Engine — Multi-strategy text injection for modern web frameworks.

Provides trusted input methods that bypass React SyntheticEvent filtering,
Lexical editor guards, and contenteditable validation layers.

Typing Strategy Waterfall:
  1. Playwright keyboard events, one character at a time with measured cadence
  2. CDP Input.dispatchKeyEvent per character (lowest-level fallback)
  3. CDP Input.insertText only for rich/contenteditable editors that require it

Each strategy includes post-type verification to confirm text was accepted.
"""

from __future__ import annotations

import asyncio
import random
from typing import Optional

from playwright.async_api import Page

from agent_first_browse.logging import get_logger

logger = get_logger("cdp_input")


# Keep genuine per-key events and jitter while centring ordinary survey text at
# roughly 80ms/character. The prior distribution averaged ~140ms and spent
# minutes typing short free-text answers during long runs.
HUMAN_KEY_INTERVALS = ((0.03, 0.07), (0.07, 0.12), (0.12, 0.20))
HUMAN_KEY_WEIGHTS = (0.30, 0.60, 0.10)


def _human_key_delay() -> float:
    interval = random.choices(HUMAN_KEY_INTERVALS, weights=HUMAN_KEY_WEIGHTS, k=1)[0]
    return random.uniform(interval[0], interval[1])


# ═══════════════════════════════════════════════════════════════════════════════
#  CDP Session Helper
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_cdp_session(page: Page):
    """Create a CDP session from a Playwright page."""
    try:
        return await page.context.new_cdp_session(page)
    except Exception as e:
        logger.warning("CDP session creation failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Post-Type Verification
# ═══════════════════════════════════════════════════════════════════════════════

async def _verify_typed_text(
    page: Page,
    expected_text: str,
    timeout: float = 3.0,
    expected_element_id: str | None = None,
) -> dict:
    """Verify that the intended field contains the expected text.

    When an element ID is available, verification is deliberately scoped to
    that exact live DOM node. A different populated input must never make a
    failed type appear successful (for example, a postcode field satisfying a
    date-of-birth verification).

    Returns dict with keys:
      - verified: bool
      - actual_length: int
      - actual_preview: str (first 100 chars)
      - match_ratio: float (0.0 to 1.0)
    """
    try:
        result = await asyncio.wait_for(page.evaluate("""
        ({ expectedId }) => {
            const read = (candidate) => {
                if (!candidate || !candidate.isConnected) return null;
                let el = candidate;
                if (el.value === undefined && !el.isContentEditable) {
                    el = el.querySelector && el.querySelector(
                        'input, textarea, [contenteditable="true"], [role="textbox"]'
                    );
                }
                if (!el || !el.isConnected) return null;
                if (el.value !== undefined) return String(el.value || '');
                if (el.isContentEditable || el.contentEditable === 'true' ||
                    el.getAttribute('role') === 'textbox') {
                    return String(el.innerText || el.textContent || '');
                }
                return null;
            };

            // Strong identity path: never fall through to another field.
            if (expectedId) {
                const target = (window.__aid || {})[expectedId];
                const text = read(target);
                return text === null
                    ? { found: false, text: '', targetMissing: true }
                    : { found: true, text };
            }

            // Legacy callers without an element ID may verify the focused field.
            const active = document.activeElement;
            if (active) {
                const text = read(active);
                if (text !== null) return { found: true, text };
            }

            // Rich editors sometimes move focus to an inner textbox.
            const candidates = document.querySelectorAll(
                'input:focus, textarea:focus, [contenteditable="true"]:focus, '
                + '[contenteditable="true"][data-lexical-editor], '
                + '[role="textbox"]'
            );
            for (const el of candidates) {
                const text = read(el);
                if (text !== null) return { found: true, text };
            }

            return { found: false, text: '' };
        }
        """, {"expectedId": expected_element_id or ""}), timeout=timeout)

        if not result.get("found"):
            return {"verified": False, "actual_length": 0, "actual_preview": "", "match_ratio": 0.0}

        actual_text = result.get("text", "")
        actual_len = len(actual_text)
        expected_len = len(expected_text)

        # Calculate match ratio
        if expected_len == 0:
            match_ratio = 1.0 if actual_len == 0 else 0.0
        else:
            # Check if the expected text is contained in or matches the actual text
            if expected_text in actual_text or actual_text in expected_text:
                match_ratio = min(actual_len, expected_len) / max(actual_len, expected_len)
            else:
                # Character-level comparison
                matching_chars = sum(1 for a, b in zip(actual_text, expected_text) if a == b)
                match_ratio = matching_chars / expected_len

        return {
            "verified": match_ratio >= 0.85,  # 85% match threshold
            "actual_length": actual_len,
            "actual_preview": actual_text[:100],
            "match_ratio": match_ratio,
        }

    except Exception as e:
        logger.warning("Type verification failed: %s", e)
        return {"verified": False, "actual_length": 0, "actual_preview": "", "match_ratio": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Rich-editor-only CDP Input.insertText fallback
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_cdp_insert_text(page: Page, text: str) -> bool:
    """Bulk-insert only into rich editors that cannot accept normal key events."""
    try:
        rich_editor = await page.evaluate("""() => {
            const el = document.activeElement;
            if (!el) return false;
            const tag = String(el.tagName || '').toLowerCase();
            return !!el.isContentEditable ||
                (el.getAttribute('role') === 'textbox' && tag !== 'input' && tag !== 'textarea');
        }""")
    except Exception:
        rich_editor = False
    if not rich_editor:
        logger.info("Skipping bulk insertText for a normal keyboard field")
        return False

    cdp = await _get_cdp_session(page)
    if not cdp:
        return False

    try:
        await cdp.send("Input.insertText", {"text": text})
        await asyncio.sleep(0.3)
        logger.debug("CDP insertText: injected %d chars", len(text))
        return True
    except Exception as e:
        logger.warning("CDP insertText failed: %s", e)
        return False
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: Playwright per-key events with human-like measured cadence
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_human_keyboard(page: Page, text: str) -> bool:
    """Send one key at a time with an independently sampled human pause.

    This generates individual keydown/keyup events that React's
    onChange handlers process one at a time. It intentionally never uses
    insert_text(), clipboard APIs, fill(), or a whole-string type operation.
    """
    try:
        for index, char in enumerate(text):
            if char == "\n":
                await page.keyboard.press("Enter")
            elif char == "\t":
                await page.keyboard.press("Tab")
            else:
                await page.keyboard.type(char, delay=0)
            await asyncio.sleep(_human_key_delay())
            if index and index % 50 == 0:
                try:
                    if page.is_closed():
                        return False
                except Exception:
                    return False
        await asyncio.sleep(0.3)
        logger.debug("Human keyboard: typed %d chars as individual key events", len(text))
        return True
    except Exception as e:
        logger.warning("Human keyboard typing failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: CDP per-character key events (Lowest level, most compatible)
# ═══════════════════════════════════════════════════════════════════════════════

async def _strategy_cdp_key_events(page: Page, text: str) -> bool:
    """Dispatch individual CDP key events per character with realistic timing.

    This is the most compatible method — generates keyDown, char, keyUp
    events that are indistinguishable from real keyboard input.
    """
    cdp = await _get_cdp_session(page)
    if not cdp:
        return False

    try:
        for i, char in enumerate(text):
            if char == '\n':
                # Enter key
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": "Enter", "code": "Enter",
                    "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": "Enter", "code": "Enter",
                    "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
                })
            elif char == '\t':
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": "Tab", "code": "Tab",
                    "windowsVirtualKeyCode": 9, "nativeVirtualKeyCode": 9,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": "Tab", "code": "Tab",
                    "windowsVirtualKeyCode": 9, "nativeVirtualKeyCode": 9,
                })
            else:
                key_code = ord(char.upper()) if char.isalpha() else ord(char)
                code_name = f"Key{char.upper()}" if char.isalpha() else ""

                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyDown", "key": char, "code": code_name,
                    "text": char,
                    "windowsVirtualKeyCode": key_code,
                    "nativeVirtualKeyCode": key_code,
                    "autoRepeat": False,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "char", "text": char,
                    "key": char, "code": code_name,
                    "windowsVirtualKeyCode": key_code,
                    "nativeVirtualKeyCode": key_code,
                })
                await cdp.send("Input.dispatchKeyEvent", {
                    "type": "keyUp", "key": char, "code": code_name,
                    "windowsVirtualKeyCode": key_code,
                    "nativeVirtualKeyCode": key_code,
                })

            # Same measured cadence as the high-level keyboard path.
            await asyncio.sleep(_human_key_delay())

            # Periodic abort check
            if i > 0 and i % 50 == 0:
                try:
                    if page.is_closed():
                        logger.warning("CDP key events aborted: page closed at char %d/%d", i, len(text))
                        return False
                except Exception:
                    return False

        logger.debug("CDP key events: dispatched %d chars", len(text))
        return True
    except Exception as e:
        logger.warning("CDP key events failed: %s", e)
        return False
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Field Clearing
# ═══════════════════════════════════════════════════════════════════════════════

async def clear_field(page: Page, element_id: str | None = None) -> None:
    """Clear the currently focused input field using multiple strategies."""
    try:
        # Strategy 1: Select all + delete
        await page.keyboard.press("Control+A")
        await asyncio.sleep(0.1)
        await page.keyboard.press("Delete")
        await asyncio.sleep(0.1)

        # Strategy 2: Verify it's actually empty, if not try Backspace
        if element_id:
            result = await _verify_typed_text(
                page, "", timeout=1.0, expected_element_id=element_id
            )
        else:
            result = await _verify_typed_text(page, "", timeout=1.0)
        if result.get("actual_length", 0) > 0:
            await page.keyboard.press("Control+A")
            await asyncio.sleep(0.05)
            await page.keyboard.press("Backspace")
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.warning("Field clear failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point: Resilient Type with Waterfall
# ═══════════════════════════════════════════════════════════════════════════════

async def _delayed_recheck(
    page: Page,
    expected_text: str,
    delay_ms: int = 300,
    element_id: str | None = None,
) -> bool:
    """Wait delay_ms, then recheck if the active field still has the expected text.

    This catches React/Vue controlled component reversions where:
      1. CDP insertText sets the DOM value immediately
      2. Our verification reads it back → PASS
      3. React's setState reconciler overwrites it 100-300ms later → FAIL

    Returns True if text is still intact, False if it was reverted.
    """
    await asyncio.sleep(delay_ms / 1000.0)
    if element_id:
        recheck = await _verify_typed_text(
            page, expected_text, timeout=2.0, expected_element_id=element_id
        )
    else:
        recheck = await _verify_typed_text(page, expected_text, timeout=2.0)
    if recheck["verified"]:
        return True
    else:
        logger.warning(
            "🔄 REACT REVERSION DETECTED: field was overwritten after %dms "
            "(now: %d chars, %.0f%% match: '%s')",
            delay_ms,
            recheck["actual_length"],
            recheck["match_ratio"] * 100,
            recheck["actual_preview"][:60],
        )
        return False


async def resilient_type(
    page: Page,
    text: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    clear_first: bool = True,
    force_retype: bool = False,
    max_retries: int = 3,
    element_id: str | None = None,
) -> dict:
    """Type text into the focused field using a multi-strategy waterfall.

    If x, y are provided, clicks to focus the field first.
    Tries increasingly aggressive strategies until text is verified.

    current UPGRADE: After initial verification, performs a delayed recheck
    (300ms) to catch React/Vue state reversions. If reverted, falls through
    to the next strategy (cdp_key_events fires individual keystrokes that
    React's onChange handler processes correctly).

    Returns dict:
      - success: bool
      - strategy: str (which strategy worked)
      - verified: bool
      - actual_length: int
      - attempts: int
      - react_stable: bool (True if text survived delayed recheck)
    """
    from agent_first_browse.browser.ghost_input import ghost_click  # Avoid circular import

    strategies = [
        ("human_keyboard", _strategy_human_keyboard),
        ("cdp_key_events", _strategy_cdp_key_events),
        ("rich_text_insert", _strategy_cdp_insert_text),
    ]

    for attempt in range(max_retries):
        strategy_name, strategy_fn = strategies[min(attempt, len(strategies) - 1)]

        logger.info(
            "Typing attempt %d/%d via %s (%d chars)",
            attempt + 1, max_retries, strategy_name, len(text),
        )

        # Focus the field if coordinates provided
        if x is not None and y is not None:
            try:
                await ghost_click(page, x, y)
                await asyncio.sleep(0.3)
                # current F2: Collapse selection to end of field value (W3C setSelectionRange)
                # React/Angular onFocus handlers often call select() which highlights all text.
                # setSelectionRange(len, len) deselects and places cursor at end, preventing
                # subsequent insertText from overwriting existing content.
                await page.evaluate("""() => {
                    const el = document.activeElement;
                    if (el && typeof el.setSelectionRange === 'function') {
                        try {
                            const len = (el.value || '').length;
                            el.setSelectionRange(len, len);
                        } catch(_) {}
                    } else if (el && el.isContentEditable) {
                        try {
                            const sel = window.getSelection();
                            if (sel && el.lastChild) {
                                sel.collapse(el.lastChild, el.lastChild.length || 0);
                            }
                        } catch(_) {}
                    }
                }""")
            except Exception as e:
                logger.warning("Focus click failed: %s", e)

        # B: Smart clear detection — auto-detect append vs replace
        if clear_first and not force_retype:
            try:
                existing_val = await page.evaluate("""() => {
                    const el = document.activeElement;
                    if (!el) return '';
                    return (el.value || el.textContent || '').trim();
                }""")
                if existing_val:
                    if text.startswith(existing_val):
                        # Agent text is a superset — just append the delta
                        delta = text[len(existing_val):]
                        if delta:
                            logger.info(
                                "Smart clear: existing '%s...' is prefix of target, appending delta '%s...' (%d chars)",
                                existing_val[:30], delta[:30], len(delta),
                            )
                            text = delta
                        else:
                            logger.info("Smart clear: field already contains exact target text — skipping type")
                            return {
                                "success": True, "strategy": "smart_skip",
                                "verified": True, "actual_length": len(existing_val),
                                "attempts": attempt + 1, "react_stable": True,
                                "no_op": True,
                            }
                        clear_first = False
                    elif existing_val.endswith(text):
                        # Target text is already at the end — skip
                        logger.info("Smart clear: target text already at end of field — skipping type")
                        return {
                            "success": True, "strategy": "smart_skip",
                            "verified": True, "actual_length": len(existing_val),
                            "attempts": attempt + 1, "react_stable": True,
                            "no_op": True,
                        }
            except Exception as e:
                logger.debug("Smart clear detection failed: %s — falling back to normal clear", e)

        # Clear existing content
        if clear_first:
            if element_id:
                await clear_field(page, element_id=element_id)
            else:
                await clear_field(page)

        # Execute the typing strategy
        ok = await strategy_fn(page, text)
        if not ok:
            logger.warning("%s execution failed — trying next strategy", strategy_name)
            continue

        # ── Immediate verification ──
        await asyncio.sleep(0.3)
        if element_id:
            verification = await _verify_typed_text(
                page, text, expected_element_id=element_id
            )
        else:
            verification = await _verify_typed_text(page, text)

        if not verification["verified"]:
            # Text wasn't accepted at all — try next strategy
            logger.warning(
                "⚠️ %s: text not fully accepted (got %d chars, %.0f%% match: '%s')",
                strategy_name,
                verification["actual_length"],
                verification["match_ratio"] * 100,
                verification["actual_preview"][:60],
            )
            continue

        # ── Delayed React reversion check ──
        # The rich-editor bulk fallback needs a delayed framework reversion
        # check. Both keyboard strategies already fire onChange per character.
        react_stable = True
        if strategy_name == "rich_text_insert":
            react_stable = await _delayed_recheck(
                page, text, delay_ms=300, element_id=element_id
            )
            if not react_stable:
                logger.warning(
                    "⚠️ %s passed immediate verify but React reverted text — "
                    "escalating to cdp_key_events",
                    strategy_name,
                )
                # Force next attempt to use cdp_key_events
                # by skipping to the last strategy index
                if attempt < max_retries - 1:
                    strategies[attempt + 1] = ("cdp_key_events", _strategy_cdp_key_events)
                continue

        logger.info(
            "✅ TYPE SUCCESS via %s: %d chars verified (%.0f%% match)%s",
            strategy_name,
            verification["actual_length"],
            verification["match_ratio"] * 100,
            " [React-stable]" if react_stable else "",
        )
        return {
            "success": True,
            "strategy": strategy_name,
            "verified": True,
            "actual_length": verification["actual_length"],
            "attempts": attempt + 1,
            "react_stable": react_stable,
        }

    # All strategies exhausted
    logger.error(
        "❌ TYPE FAILED: all %d strategies exhausted for %d-char text",
        max_retries, len(text),
    )
    return {
        "success": False,
        "strategy": "none",
        "verified": False,
        "actual_length": 0,
        "attempts": max_retries,
        "react_stable": False,
    }
