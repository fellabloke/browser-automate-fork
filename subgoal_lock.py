"""Compatibility alias for the canonical agent_first_browse.cognition.subgoal_lock module."""

import sys

from agent_first_browse.cognition import subgoal_lock as _module

sys.modules[__name__] = _module
