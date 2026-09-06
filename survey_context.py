"""Compatibility alias for the canonical context module."""

import sys

from agent_first_browse.survey import context as _module

sys.modules[__name__] = _module
