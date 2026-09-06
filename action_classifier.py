"""Compatibility alias for the canonical agent_first_browse.cognition.action_classifier module."""

import sys

from agent_first_browse.cognition import action_classifier as _module

sys.modules[__name__] = _module
