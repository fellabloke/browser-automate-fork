"""Compatibility alias for the canonical agent_first_browse.verification.overwatch module."""

import sys

from agent_first_browse.verification import overwatch as _module

sys.modules[__name__] = _module
