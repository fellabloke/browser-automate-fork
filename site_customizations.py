"""Narrow, hostname-scoped page customizations used by the browser agent."""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


SITE_CUSTOMIZATION_INIT_SCRIPT = r"""
(() => {
    const hostname = String(window.location.hostname || '').toLowerCase();
    const isSurveyStreak = hostname.split('.').some(
        label => label === 'surveystreak' || label === 'survey-streak'
    );
    if (!isSurveyStreak) return;

    const styleId = 'agent-surveystreak-customizations';
    const install = () => {
        if (document.getElementById(styleId)) return;
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            a.games-offer-banner,
            .games-offer-banner {
                display: none !important;
                visibility: hidden !important;
                pointer-events: none !important;
            }
        `;
        (document.head || document.documentElement).appendChild(style);
    };

    if (document.documentElement) install();
    else document.addEventListener('DOMContentLoaded', install, { once: true });
})();
"""


async def install_site_customizations(context: Any, page: Any | None = None) -> None:
    """Install customizations for future documents and the current CDP page."""
    await context.add_init_script(SITE_CUSTOMIZATION_INIT_SCRIPT)
    if page is not None:
        try:
            await page.evaluate(SITE_CUSTOMIZATION_INIT_SCRIPT)
        except Exception as exc:  # Cross-process navigation can race attachment.
            logger.debug("Current-page site customization deferred to navigation: %s", exc)


async def apply_current_site_customizations(page: Any) -> None:
    """Apply to an already-loaded page without registering another init script."""
    try:
        await page.evaluate(SITE_CUSTOMIZATION_INIT_SCRIPT)
    except Exception as exc:
        logger.debug("Current-page site customization deferred to navigation: %s", exc)
