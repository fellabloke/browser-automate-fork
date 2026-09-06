"""Compatibility alias for the canonical google_stealth_auth_graph module."""

import sys

from agent_first_browse.promotion.browser_promoter import google_stealth_auth_graph as _module

sys.modules[__name__] = _module
