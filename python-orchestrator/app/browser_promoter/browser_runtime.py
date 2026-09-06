"""Compatibility alias for the canonical browser_runtime module."""

import sys

from agent_first_browse.promotion.browser_promoter import browser_runtime as _module

sys.modules[__name__] = _module
