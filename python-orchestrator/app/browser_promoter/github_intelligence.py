"""Compatibility alias for the canonical github_intelligence module."""

import sys

from agent_first_browse.promotion.browser_promoter import github_intelligence as _module

sys.modules[__name__] = _module
