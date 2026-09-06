"""Compatibility alias for the canonical agent_first_browse.verification.action module."""

import sys

from agent_first_browse.verification import action as _module

sys.modules[__name__] = _module
