"""Compatibility alias for the canonical agent_first_browse.cognition.content_critic module."""

import sys

from agent_first_browse.cognition import content_critic as _module

sys.modules[__name__] = _module
