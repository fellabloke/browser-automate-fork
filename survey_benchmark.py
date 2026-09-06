"""Compatibility alias for the canonical benchmarks module."""

import sys

from agent_first_browse.survey import benchmarks as _module

sys.modules[__name__] = _module
