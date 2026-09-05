"""Regression tests for popup adoption and human per-key survey input."""

from __future__ import annotations

import pytest

import brain_graph
from agent_first_browse.browser import cdp_input
import mcp_tools


class _Keyboard:
    def __init__(self):
        self.typed: list[tuple[str, int]] = []
        self.pressed: list[str] = []

    async def type(self, text: str, delay: int = 0):
        self.typed.append((text, delay))

    async def press(self, key: str):
        self.pressed.append(key)


class _TypingPage:
    def __init__(self, evaluate_result=False):
        self.keyboard = _Keyboard()
        self._evaluate_result = evaluate_result

    async def evaluate(self, _script):
        return self._evaluate_result

    def is_closed(self):
        return False


@pytest.mark.asyncio
async def test_human_keyboard_sends_individual_keys(monkeypatch):
    page = _TypingPage()
    sleeps: list[float] = []

    async def no_wait(delay):
        sleeps.append(delay)

    monkeypatch.setattr(cdp_input.asyncio, "sleep", no_wait)
    monkeypatch.setattr(cdp_input, "_human_key_delay", lambda: 0.123)

    assert await cdp_input._strategy_human_keyboard(page, "Ab 3\n")
    assert page.keyboard.typed == [("A", 0), ("b", 0), (" ", 0), ("3", 0)]
    assert page.keyboard.pressed == ["Enter"]
    assert sleeps[:5] == [0.123] * 5
    assert all(len(text) == 1 for text, _ in page.keyboard.typed)


def test_human_key_delay_uses_reference_distribution(monkeypatch):
    monkeypatch.setattr(
        cdp_input.random,
        "choices",
        lambda intervals, weights, k: [intervals[1]],
    )
    monkeypatch.setattr(cdp_input.random, "uniform", lambda low, high: (low + high) / 2)

    assert cdp_input._human_key_delay() == pytest.approx(0.095)
    assert pytest.approx((0.30, 0.60, 0.10)) == cdp_input.HUMAN_KEY_WEIGHTS


@pytest.mark.asyncio
async def test_bulk_insert_is_refused_for_normal_input(monkeypatch):
    page = _TypingPage(evaluate_result=False)
    cdp_requested = False

    async def should_not_request_cdp(_page):
        nonlocal cdp_requested
        cdp_requested = True
        return None

    monkeypatch.setattr(cdp_input, "_get_cdp_session", should_not_request_cdp)

    assert await cdp_input._strategy_cdp_insert_text(page, "normal answer") is False
    assert cdp_requested is False


class _Context:
    def __init__(self):
        self.pages: list[_TabPage] = []

    async def new_page(self):
        page = _TabPage(self, "about:blank")
        self.pages.append(page)
        return page


class _TabPage:
    def __init__(self, context, url: str, opener=None):
        self.context = context
        self.url = url
        self._opener = opener
        self._closed = False
        self.fronted = False

    def is_closed(self):
        return self._closed

    async def opener(self):
        return self._opener

    async def wait_for_load_state(self, _state, timeout):
        return None

    async def bring_to_front(self):
        self.fronted = True

    async def goto(self, url, **_kwargs):
        self.url = url

    async def close(self):
        self._closed = True


@pytest.mark.asyncio
async def test_new_popup_is_adopted_as_active_page():
    context = _Context()
    dashboard = _TabPage(context, "https://example.test/dashboard")
    context.pages.append(dashboard)
    mcp_tools.set_page(dashboard)

    survey = _TabPage(context, "https://survey-provider.test/question/1", opener=dashboard)
    context.pages.append(survey)

    result = await mcp_tools.adopt_new_page_if_opened(dashboard, wait_ms=10)

    assert result["switched"] is True
    assert result["reason"] == "new_popup"
    assert mcp_tools.get_page() is survey
    assert survey.fronted is True


@pytest.mark.asyncio
async def test_brain_perception_and_critic_follow_adopted_popup():
    context = _Context()
    dashboard = _TabPage(context, "https://example.test/dashboard")
    context.pages.append(dashboard)
    mcp_tools.set_page(dashboard)
    brain_graph._PAGE = dashboard

    class _Critic:
        _page = dashboard

    critic = _Critic()
    brain_graph._CRITIC = critic
    survey = _TabPage(context, "https://survey-provider.test/question/1", opener=dashboard)
    context.pages.append(survey)

    active = await brain_graph._sync_active_page()

    assert active is survey
    assert brain_graph._PAGE is survey
    assert critic._page is survey
    assert mcp_tools.get_page() is survey


@pytest.mark.asyncio
async def test_closed_survey_tab_returns_to_live_opener():
    context = _Context()
    dashboard = _TabPage(context, "https://example.test/dashboard")
    survey = _TabPage(context, "https://survey-provider.test/complete", opener=dashboard)
    context.pages.extend([dashboard, survey])
    mcp_tools.set_page(survey)
    survey._closed = True

    result = await mcp_tools.adopt_new_page_if_opened(survey, wait_ms=10)

    assert result["switched"] is True
    assert result["reason"] == "active_closed"
    assert mcp_tools.get_page() is dashboard


@pytest.mark.asyncio
async def test_fresh_dashboard_mode_creates_new_tab_and_closes_stale_qmee_tabs():
    context = _Context()
    dashboard = _TabPage(context, "https://www.qmee.com/en-gb/surveys")
    survey = _TabPage(context, "https://provider.test/complete", opener=dashboard)
    context.pages.extend([dashboard, survey])
    mcp_tools.set_page(survey)

    result = await mcp_tools.mcp_abandon_survey(
        "https://www.qmee.com/en-gb/surveys", fresh_dashboard=True
    )

    fresh = mcp_tools.get_page()
    assert result["success"] is True
    assert fresh not in {dashboard, survey}
    assert fresh.url == "https://www.qmee.com/en-gb/surveys"
    assert dashboard.is_closed() and survey.is_closed()
