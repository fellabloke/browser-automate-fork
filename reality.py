"""Compatibility alias for the canonical agent_first_browse.cognition.reality module."""

import sys

from agent_first_browse.cognition import reality as _module

sys.modules[__name__] = _module
