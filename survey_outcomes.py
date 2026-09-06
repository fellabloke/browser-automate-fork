"""Compatibility alias for the canonical outcomes module."""

import sys

from agent_first_browse.survey import outcomes as _module

sys.modules[__name__] = _module
