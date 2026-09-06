"""Compatibility alias for the canonical browser_warmup module."""

import sys

from agent_first_browse.promotion.browser_promoter import browser_warmup as _module

sys.modules[__name__] = _module
