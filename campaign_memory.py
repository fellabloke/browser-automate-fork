"""Compatibility alias for the canonical campaign module."""

import sys

from agent_first_browse.memory import campaign as _module

sys.modules[__name__] = _module
