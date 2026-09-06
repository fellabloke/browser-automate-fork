"""Regression tests for hostname-scoped SurveyStreak customizations."""

from __future__ import annotations

import asyncio

from agent_first_browse.browser.site_customizations import (
    SITE_CUSTOMIZATION_INIT_SCRIPT,
    apply_current_site_customizations,
    install_site_customizations,
)


class _FakeContext:
    def __init__(self):
        self.scripts = []

    async def add_init_script(self, script):
        self.scripts.append(script)


class _FakePage:
    def __init__(self):
        self.scripts = []

    async def evaluate(self, script):
        self.scripts.append(script)


def test_surveystreak_banner_rule_is_hostname_scoped_and_strong():
    script = SITE_CUSTOMIZATION_INIT_SCRIPT

    assert "window.location.hostname" in script
    assert "label === 'surveystreak'" in script
    assert "label === 'survey-streak'" in script
    assert ".games-offer-banner" in script
    assert "display: none !important" in script
    assert "pointer-events: none !important" in script
    assert "if (!isSurveyStreak) return" in script


def test_install_registers_future_documents_and_updates_current_page():
    context = _FakeContext()
    page = _FakePage()

    asyncio.run(install_site_customizations(context, page))

    assert context.scripts == [SITE_CUSTOMIZATION_INIT_SCRIPT]
    assert page.scripts == [SITE_CUSTOMIZATION_INIT_SCRIPT]


def test_current_page_apply_does_not_register_duplicate_init_script():
    page = _FakePage()

    asyncio.run(apply_current_site_customizations(page))

    assert page.scripts == [SITE_CUSTOMIZATION_INIT_SCRIPT]
