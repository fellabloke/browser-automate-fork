"""Compatibility alias for the canonical agent_first_browse.verification.engine module."""

import sys

from agent_first_browse.verification import engine as _module

sys.modules[__name__] = _module
