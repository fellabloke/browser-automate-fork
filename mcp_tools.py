"""Compatibility alias for the canonical agent_first_browse.actions.tools module."""

import sys

from agent_first_browse.actions import tools as _module

sys.modules[__name__] = _module
