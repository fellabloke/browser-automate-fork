"""Compatibility alias for the canonical cdp_stealth_launcher module."""

import sys

from agent_first_browse.promotion.browser_promoter import cdp_stealth_launcher as _module

sys.modules[__name__] = _module
