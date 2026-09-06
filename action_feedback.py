"""Compatibility alias for the canonical agent_first_browse.verification.feedback module."""

import sys

from agent_first_browse.verification import feedback as _module

sys.modules[__name__] = _module
