"""Compatibility alias for the canonical shadow_dom_piercer module."""

import sys

from agent_first_browse.promotion.browser_promoter import shadow_dom_piercer as _module

sys.modules[__name__] = _module
