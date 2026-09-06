"""Compatibility alias for the canonical state module."""

import sys

from agent_first_browse.promotion.browser_promoter import state as _module

sys.modules[__name__] = _module
