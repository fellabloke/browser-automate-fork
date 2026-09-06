"""Compatibility alias for the canonical dashboard module."""

import sys

from agent_first_browse.promotion.browser_promoter import dashboard as _module

sys.modules[__name__] = _module
