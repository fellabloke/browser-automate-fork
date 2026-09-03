"""Regression tests for semantic actions that omit vision coordinates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.browser_promoter import nodes
from app.browser_promoter.state import BrowserAction, HighLevelCommand
from app.browser_promoter.worker_planner import ReasoningDecision


@pytest.mark.asyncio
async def test_dom_fallback_resolves_fresh_element_center(monkeypatch, agent_state):
    class FakePage:
        pass

    async def fake_page(**_kwargs):
        return FakePage()

    snapshot = {
        "elements": [
            {"id": "e7", "kind": "button", "text": "Submit application"},
            {"id": "e8", "kind": "button", "text": "Cancel"},
        ]
    }

    async def fake_extract(page, target_hint=None, timeout=0):
        assert isinstance(page, FakePage)
        assert target_hint == "Submit application"
        return snapshot

    async def fake_resolve(page, eid):
        assert isinstance(page, FakePage)
        assert eid == "e7"
        return {"ok": True, "x": 321.5, "y": 204.0}

    monkeypatch.setattr(nodes.BrowserRuntime, "ensure_page", fake_page)
    monkeypatch.setitem(__import__("sys").modules, "dom_parser", SimpleNamespace(
        extract=fake_extract,
        resolve_element=fake_resolve,
    ))

    action = BrowserAction(action="click")
    grounded = await nodes._resolve_dom_fallback_action(
        action,
        state=agent_state,
        target_hint="Submit application",
    )

    assert grounded is not None
    assert grounded.x == 321.5
    assert grounded.y == 204.0


@pytest.mark.asyncio
async def test_selector_fallback_is_used_by_executor(monkeypatch):
    import app.browser_promoter.shadow_dom_piercer as piercer
    import app.browser_promoter.zero_token_executor as executor_module

    class FakePage:
        pass

    async def fake_locate(page, *, selector, text_hint=""):
        assert isinstance(page, FakePage)
        assert selector == "#submit"
        return piercer.TargetPoint(10.0, 20.0, "native-shadow-piercer", selector)

    clicked = {}

    async def fake_click(page, x, y):
        clicked.update({"page": page, "x": x, "y": y})

    monkeypatch.setattr(piercer, "locate_target_point", fake_locate)
    monkeypatch.setattr(executor_module, "ghost_click", fake_click)

    action = BrowserAction(action="click", selector="#submit")
    result = await executor_module.ZeroTokenActionExecutor().execute_action(
        page=FakePage(), action=action
    )

    assert result.details["x"] == 10.0
    assert result.details["y"] == 20.0
    assert clicked["x"] == 10.0
    assert clicked["y"] == 20.0


@pytest.mark.asyncio
async def test_second_missing_coordinate_failure_stops_loop(monkeypatch, agent_state):
    decision = ReasoningDecision(
        action=BrowserAction(action="click"),
        confidence=0.8,
        reasoning="Click Submit application",
    )

    class FakeReasoningAgent:
        async def decide_action(self, **_kwargs):
            return decision

    async def no_dom_fallback(*_args, **_kwargs):
        return None

    monkeypatch.setattr(nodes, "_get_reasoning_agent", lambda: FakeReasoningAgent())
    monkeypatch.setattr(nodes, "_resolve_dom_fallback_action", no_dom_fallback)

    state = agent_state.model_copy(update={
        "high_level_command": HighLevelCommand(
            action_type="engage",
            target_description="Submit application",
            behavior_plan="Submit the form.",
        ),
        "ephemeral": {"dom_fallback_attempts": 1},
    })

    update = await nodes.reasoning_agent_node(state)

    assert update["routing"].stop_requested is True
    assert update["worker_last_confused"] is False
    assert "vision loop" in update["worker_last_confusion_reason"]
