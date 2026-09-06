"""Compatibility alias for the canonical site_quirks module."""

import sys

from agent_first_browse.survey import site_quirks as _module

sys.modules[__name__] = _module
