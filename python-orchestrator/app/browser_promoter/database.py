"""Compatibility alias for the canonical database module."""

import sys

from agent_first_browse.promotion.browser_promoter import database as _module

sys.modules[__name__] = _module
