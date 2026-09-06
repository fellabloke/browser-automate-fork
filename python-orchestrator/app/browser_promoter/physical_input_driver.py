"""Compatibility alias for the canonical physical_input_driver module."""

import sys

from agent_first_browse.promotion.browser_promoter import physical_input_driver as _module

sys.modules[__name__] = _module
