"""Compatibility alias for the canonical profile module."""

import sys

from agent_first_browse.survey import profile as _module

sys.modules[__name__] = _module
