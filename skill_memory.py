"""Compatibility alias for the canonical skills module."""

import sys

from agent_first_browse.memory import skills as _module

sys.modules[__name__] = _module
