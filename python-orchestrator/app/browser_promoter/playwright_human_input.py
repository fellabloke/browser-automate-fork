"""Compatibility alias for the canonical playwright_human_input module."""

import sys

from agent_first_browse.promotion.browser_promoter import playwright_human_input as _module

sys.modules[__name__] = _module
