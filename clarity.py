"""Compatibility alias for the canonical agent_first_browse.cognition.clarity module."""

import sys

from agent_first_browse.cognition import clarity as _module

sys.modules[__name__] = _module
