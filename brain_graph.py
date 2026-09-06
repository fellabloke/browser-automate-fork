"""Compatibility façade for the canonical LangGraph orchestration module."""

import sys

from agent_first_browse.agent import graph as _graph

sys.modules[__name__] = _graph
