"""Compatibility alias for the canonical agent_first_browse.cognition.dreamer module."""

import sys

from agent_first_browse.cognition import dreamer as _module

sys.modules[__name__] = _module
