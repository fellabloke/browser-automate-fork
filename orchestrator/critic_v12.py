"""Compatibility alias for the canonical agent_first_browse.verification.progress module."""

import sys

from agent_first_browse.verification import progress as _module

sys.modules[__name__] = _module
