"""Compatibility alias for the canonical marketing_engine module."""

import sys

from agent_first_browse.promotion.browser_promoter import marketing_engine as _module

sys.modules[__name__] = _module
