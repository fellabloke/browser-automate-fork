"""Compatibility alias for the canonical intent_journal module."""

import sys

from agent_first_browse.memory import intent_journal as _module

sys.modules[__name__] = _module
