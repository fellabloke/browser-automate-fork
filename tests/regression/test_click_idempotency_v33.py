"""Regression tests for at-most-once browser clicks.

The AttaPoll failure was a custom DIV option being toggled repeatedly by the
click-strategy waterfall because its CSS-only selected state was not detected.
"""

from __future__ import annotations

from agent_first_browse.browser import cdp_click
from agent_first_browse.browser import cdp_input
import intent_journal
import mcp_tools
import overwatch
from survey_context import survey_gate_violation


def _fp(url: str = "https://example.test/survey") -> dict:
    return {
        "url": url,
        "title": "Survey",
        "element_count": 10,
        "structural_hash": "same",
    }


def test_custom_control_class_and_style_changes_are_detected():
    pre = {
        "found": True,
        "signals": [{
            "key": "target",
            "classList": "answer",
            "visualStyle": "rgb(255, 255, 255)|rgb(0, 0, 0)|none|rgb(0, 0, 0)",
        }],
    }
    post = {
        "found": True,
        "signals": [{
            "key": "target",
            "classList": "answer selected",
            "visualStyle": "rgb(20, 90, 200)|rgb(0, 0, 0)|none|rgb(0, 0, 0)",
        }],
    }

    changed, details = cdp_click._element_state_changed(pre, post)

    assert changed is True
    assert any("classList" in detail for detail in details)


def test_cosmetic_style_change_alone_is_not_selection_progress():
    pre = {
        "found": True,
        "ariaSelected": "true",
        "signals": [{
            "key": "target",
            "ariaSelected": "true",
            "classList": "answer selected",
            "visualStyle": "rgb(255, 255, 255)|rgb(0, 0, 0)|none",
        }],
    }
    post = {
        "found": True,
        "ariaSelected": "true",
        "signals": [{
            "key": "target",
            "ariaSelected": "true",
            "classList": "answer selected",
            "visualStyle": "oklab(0.98 0.01 0.01)|rgb(0, 0, 0)|none",
        }],
    }

    changed, details = cdp_click._element_state_changed(pre, post)

    assert changed is False
    assert details == []


def test_selected_checkbox_alias_blocks_label_toggle():
    selector_map = {
        "e5": {
            "kind": "button", "tag": "LABEL", "text": "Califia Farms",
            "x": 472, "y": 226,
        },
        "e6": {
            "kind": "input", "tag": "DIV", "text": "", "x": 471, "y": 226,
            "control_type": "checkbox", "selected": True,
        },
        "e7": {"kind": "button", "text": "Next", "x": 805, "y": 384},
    }

    violation = survey_gate_violation(
        {"verb": "click", "element_id": "e5"}, selector_map,
        page_text="Select all brands that apply", continuous_mode=True,
    )

    assert "already selected" in violation
    assert "undo" in violation


async def test_js_fallback_emits_exactly_one_click_event():
    class Page:
        script = ""

        async def evaluate(self, script, arg):
            self.script = script
            return {"clicked": True, "tag": "div", "text": "Tesco"}

    page = Page()
    assert await cdp_click._strategy_js_click(page, 100, 200) is True
    assert page.script.count("el.click();") == 1
    assert "new MouseEvent('click'" not in page.script


async def test_exact_non_link_is_dispatched_at_most_once(monkeypatch):
    calls: list[str] = []

    class Page:
        async def wait_for_load_state(self, *args, **kwargs):
            return None

    async def fingerprint(page):
        return _fp()

    async def state(page, x, y, timeout=2.0, element_id=None):
        return {
            "found": True,
            "exactTarget": True,
            "interactionKind": "control",
            "classList": "answer",
            "signals": [{"key": "target", "classList": "answer"}],
        }

    async def native(*args, **kwargs):
        calls.append("native")
        return True

    async def should_not_run(*args, **kwargs):
        calls.append("fallback")
        return True

    monkeypatch.setattr(cdp_click, "_capture_page_fingerprint", fingerprint)
    monkeypatch.setattr(cdp_click, "_capture_element_state", state)
    monkeypatch.setattr(cdp_click, "_strategy_cdp_mouse_event", native)
    monkeypatch.setattr(cdp_click, "_strategy_js_click", should_not_run)
    monkeypatch.setattr(cdp_click, "_strategy_direct_navigate", should_not_run)
    monkeypatch.setattr(cdp_click, "_strategy_playwright_click", should_not_run)

    result = await cdp_click.resilient_click(
        Page(), 100, 200, settle_ms=0, element_id="e1"
    )

    assert calls == ["native"]
    assert result.success is True
    assert result.dispatched is True
    assert result.verified is False
    assert result.attempts == 1
    assert "single_dispatch" in result.strategy


async def test_exact_custom_state_change_stops_after_native_click(monkeypatch):
    calls: list[str] = []
    snapshots = iter([
        {
            "found": True,
            "exactTarget": True,
            "interactionKind": "control",
            "signals": [{"key": "target", "classList": "answer"}],
        },
        {
            "found": True,
            "exactTarget": True,
            "interactionKind": "control",
            "signals": [{"key": "target", "classList": "answer selected"}],
        },
    ])

    class Page:
        async def wait_for_load_state(self, *args, **kwargs):
            return None

    async def fingerprint(page):
        return _fp()

    async def state(page, x, y, timeout=2.0, element_id=None):
        return next(snapshots)

    async def native(*args, **kwargs):
        calls.append("native")
        return True

    async def should_not_run(*args, **kwargs):
        calls.append("fallback")
        return True

    monkeypatch.setattr(cdp_click, "_capture_page_fingerprint", fingerprint)
    monkeypatch.setattr(cdp_click, "_capture_element_state", state)
    monkeypatch.setattr(cdp_click, "_strategy_cdp_mouse_event", native)
    monkeypatch.setattr(cdp_click, "_strategy_js_click", should_not_run)
    monkeypatch.setattr(cdp_click, "_strategy_direct_navigate", should_not_run)
    monkeypatch.setattr(cdp_click, "_strategy_playwright_click", should_not_run)

    result = await cdp_click.resilient_click(
        Page(), 100, 200, settle_ms=0, element_id="e1"
    )

    assert calls == ["native"]
    assert result.success is True
    assert result.verified is True
    assert result.attempts == 1
    assert "state_verified" in result.strategy


async def test_live_selected_preflight_suppresses_answer_toggle(monkeypatch):
    calls: list[str] = []

    class Page:
        pass

    async def fingerprint(page):
        return _fp()

    async def selected_state(*args, **kwargs):
        return {
            "found": True,
            "exactTarget": True,
            "interactionKind": "control",
            "dataSelected": "true",
            "signals": [{"key": "control", "checked": True}],
        }

    async def must_not_click(*args, **kwargs):
        calls.append("clicked")
        return True

    monkeypatch.setattr(cdp_click, "_capture_page_fingerprint", fingerprint)
    monkeypatch.setattr(cdp_click, "_capture_element_state", selected_state)
    monkeypatch.setattr(cdp_click, "_strategy_cdp_mouse_event", must_not_click)

    result = await cdp_click.resilient_click(
        Page(), 472, 226, element_id="e5", prevent_deselect=True,
    )

    assert calls == []
    assert result.no_op is True
    assert result.dispatched is False
    assert result.verified is False


async def test_link_keeps_direct_navigation_fallback(monkeypatch):
    calls: list[str] = []
    fingerprints = iter([
        _fp("https://example.test/start"),
        _fp("https://example.test/start"),
        _fp("https://example.test/destination"),
    ])

    class Page:
        async def wait_for_load_state(self, *args, **kwargs):
            return None

    async def fingerprint(page):
        return next(fingerprints)

    async def state(page, x, y, timeout=2.0, element_id=None):
        return {
            "found": True,
            "exactTarget": True,
            "interactionKind": "link",
            "signals": [{"key": "target", "classList": "nav-link"}],
        }

    async def native(*args, **kwargs):
        calls.append("native")
        return True

    async def direct(*args, **kwargs):
        calls.append("direct")
        return True

    async def should_not_run(*args, **kwargs):
        calls.append("unsafe-fallback")
        return True

    monkeypatch.setattr(cdp_click, "_capture_page_fingerprint", fingerprint)
    monkeypatch.setattr(cdp_click, "_capture_element_state", state)
    monkeypatch.setattr(cdp_click, "_strategy_cdp_mouse_event", native)
    monkeypatch.setattr(cdp_click, "_strategy_direct_navigate", direct)
    monkeypatch.setattr(cdp_click, "_strategy_js_click", should_not_run)
    monkeypatch.setattr(cdp_click, "_strategy_playwright_click", should_not_run)

    result = await cdp_click.resilient_click(
        Page(), 100, 200, settle_ms=0, element_id="e1"
    )

    assert calls == ["native", "direct"]
    assert result.success is True
    assert result.navigation is True
    assert result.strategy == "direct_navigate"


async def test_unverified_dispatch_stays_pending_in_action_journal(monkeypatch):
    async def click_once(**kwargs):
        return {
            "success": True,
            "strategy": "cdp_mouse_event+single_dispatch_unverified",
            "navigated": False,
            "dom_changed": False,
            "verified": False,
            "error": "",
        }

    monkeypatch.setattr(mcp_tools, "mcp_click", click_once)
    outcome = await overwatch._execute_action(
        {"verb": "click", "element_id": "e1", "x": 100, "y": 200}, None
    )

    assert "DISPATCHED ONCE" in outcome
    assert "verification pending" in outcome
    assert "OK" not in outcome
    assert intent_journal.classify_status(outcome) == "executed"


async def test_selected_answer_click_noop_is_not_reported_as_progress(monkeypatch):
    captured = {}

    async def selected_no_op(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "strategy": "selected_state_no_op",
            "navigated": False,
            "dom_changed": False,
            "verified": False,
            "no_op": True,
            "error": "",
        }

    monkeypatch.setattr(mcp_tools, "mcp_click", selected_no_op)
    outcome = await overwatch._execute_action({
        "verb": "click", "element_id": "e5", "target_name": "Califia Farms",
        "answer_basis": "configured_profile_fact",
    }, None)

    assert captured["prevent_deselect"] is True
    assert outcome.startswith("→ NO-OP")
    assert not overwatch._action_execution_confirmed(outcome)


async def test_fragment_churn_with_validation_error_is_not_navigation_progress(monkeypatch):
    calls: list[str] = []
    fingerprints = iter([
        {
            **_fp("https://survey.test/form#before"),
            "structural_hash": "before",
            "validation_error": "",
        },
        {
            **_fp("https://survey.test/form#after"),
            "structural_hash": "after",
            "validation_error": "There were problems with some data entered",
        },
    ])

    class Page:
        async def wait_for_load_state(self, *args, **kwargs):
            return None

    async def fingerprint(page):
        return next(fingerprints)

    async def state(page, x, y, timeout=2.0, element_id=None):
        return {
            "found": True,
            "exactTarget": True,
            "interactionKind": "control",
            "signals": [{"key": "target", "classList": "continue"}],
        }

    async def native(*args, **kwargs):
        calls.append("native")
        return True

    monkeypatch.setattr(cdp_click, "_capture_page_fingerprint", fingerprint)
    monkeypatch.setattr(cdp_click, "_capture_element_state", state)
    monkeypatch.setattr(cdp_click, "_strategy_cdp_mouse_event", native)

    result = await cdp_click.resilient_click(
        Page(), 100, 200, settle_ms=0, element_id="e1"
    )

    assert calls == ["native"]
    assert result.navigation is False
    assert result.verified is False
    assert "single_dispatch" in result.strategy


async def test_force_retype_flag_reaches_typing_verifier(monkeypatch):
    captured = {}

    async def type_once(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "strategy": "human_keyboard",
            "actual_length": 3,
            "error": "",
        }

    monkeypatch.setattr(mcp_tools, "mcp_type", type_once)
    outcome = await overwatch._execute_action({
        "verb": "type",
        "element_id": "e2",
        "text": "KY8",
        "force_retype": True,
    }, None)

    assert outcome.startswith("→ OK")
    assert captured["force_retype"] is True


async def test_already_correct_type_is_not_reported_as_progress(monkeypatch):
    async def no_op_type(**kwargs):
        return {
            "success": True,
            "strategy": "smart_skip",
            "actual_length": 2,
            "no_op": True,
            "error": "",
        }

    monkeypatch.setattr(mcp_tools, "mcp_type", no_op_type)
    outcome = await overwatch._execute_action({
        "verb": "type", "element_id": "e1", "text": "20",
    }, None)

    assert outcome.startswith("→ NO-OP")
    assert not overwatch._action_execution_confirmed(outcome)


async def test_force_retype_clears_filled_field_and_runs_text_verification(monkeypatch):
    events: list[str] = []

    class Page:
        pass

    async def clear(page):
        events.append("clear")
        return True

    async def type_text(page, text):
        events.append(f"type:{text}")
        return True

    async def verify(page, expected, timeout=3.0):
        events.append(f"verify:{expected}")
        return {
            "verified": True,
            "actual_length": len(expected),
            "match_ratio": 1.0,
            "actual_preview": expected,
        }

    monkeypatch.setattr(cdp_input, "clear_field", clear)
    monkeypatch.setattr(cdp_input, "_strategy_human_keyboard", type_text)
    monkeypatch.setattr(cdp_input, "_verify_typed_text", verify)

    result = await cdp_input.resilient_type(
        Page(), "KY8", clear_first=True, force_retype=True, max_retries=1
    )

    assert result["verified"] is True
    assert events == ["clear", "type:KY8", "verify:KY8"]
