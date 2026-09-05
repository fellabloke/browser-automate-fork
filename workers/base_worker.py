"""Temporary compatibility alias for the packaged worker implementation."""

import sys

from agent_first_browse.workers import base as _base

sys.modules[__name__] = _base
