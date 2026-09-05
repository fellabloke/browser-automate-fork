"""Temporary compatibility alias for the packaged model registry."""

import sys

from agent_first_browse.models import registry as _registry

sys.modules[__name__] = _registry
