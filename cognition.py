"""Compatibility alias for the canonical agent_first_browse.cognition.reasoning module."""

import sys

from agent_first_browse.cognition import reasoning as _module

sys.modules[__name__] = _module
