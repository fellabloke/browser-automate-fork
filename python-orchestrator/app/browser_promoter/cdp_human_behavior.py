"""Compatibility alias for the canonical cdp_human_behavior module."""

import sys

from agent_first_browse.promotion.browser_promoter import cdp_human_behavior as _module

sys.modules[__name__] = _module
