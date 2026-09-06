"""Compatibility alias for the canonical recipes module."""

import sys

from agent_first_browse.survey import recipes as _module

sys.modules[__name__] = _module
