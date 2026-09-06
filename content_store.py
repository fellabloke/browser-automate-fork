"""Compatibility alias for the canonical content_store module."""

import sys

from agent_first_browse.memory import content_store as _module

sys.modules[__name__] = _module
