"""Compatibility alias for the canonical agent_first_browse.cognition.prm module."""

import sys

from agent_first_browse.cognition import prm as _module

sys.modules[__name__] = _module
