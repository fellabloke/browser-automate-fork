"""Compatibility alias for the canonical agent_first_browse.cognition.core module."""

import sys

from agent_first_browse.cognition import core as _module

sys.modules[__name__] = _module
