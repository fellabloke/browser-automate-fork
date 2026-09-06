"""Compatibility alias for the canonical checkpoint retention module."""

import sys

from agent_first_browse.persistence import checkpoint_retention as _module

sys.modules[__name__] = _module
