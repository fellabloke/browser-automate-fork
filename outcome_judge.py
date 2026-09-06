"""Compatibility alias for the canonical agent_first_browse.verification.outcome module."""

import sys

from agent_first_browse.verification import outcome as _module

sys.modules[__name__] = _module
