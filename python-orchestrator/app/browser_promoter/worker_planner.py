"""Compatibility alias for the canonical worker_planner module."""

import sys

from agent_first_browse.promotion.browser_promoter import worker_planner as _module

sys.modules[__name__] = _module
