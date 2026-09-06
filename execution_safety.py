"""Compatibility alias for the canonical agent_first_browse.verification.safety module."""

import sys

from agent_first_browse.verification import safety as _module

sys.modules[__name__] = _module
