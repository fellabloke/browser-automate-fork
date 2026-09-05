"""Compatibility alias for the canonical survey audio module."""

import sys

from agent_first_browse.survey import audio as _module

sys.modules[__name__] = _module
