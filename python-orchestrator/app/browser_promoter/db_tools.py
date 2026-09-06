"""Compatibility alias for the canonical db_tools module."""

import sys

from agent_first_browse.promotion.browser_promoter import db_tools as _module

sys.modules[__name__] = _module
