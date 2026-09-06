"""Compatibility alias for the canonical supervisor_subgraph module."""

import sys

from agent_first_browse.promotion.browser_promoter import supervisor_subgraph as _module

sys.modules[__name__] = _module
