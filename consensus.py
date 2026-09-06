"""Compatibility alias for the canonical agent_first_browse.cognition.consensus module."""

import sys

from agent_first_browse.cognition import consensus as _module

sys.modules[__name__] = _module
