"""Compatibility alias for the canonical graph module."""

import sys

from agent_first_browse.promotion.browser_promoter import graph as _module

sys.modules[__name__] = _module
