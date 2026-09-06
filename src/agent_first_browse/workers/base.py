"""Base Worker — Shared LLM invocation logic for all specialist workers.

All workers share the same pattern:
  1. Build a specialist system prompt
  2. Build a user prompt with current DOM + context
  3. Invoke the LLM via the failover chain
  4. Parse the structured output into a ProposedAction
  5. Return state updates (proposed_action, history entry, etc.)

Workers NEVER execute actions directly. They propose actions that
Overwatch validates before committing.

Includes "Look-Before-You-Leap" coordinate validation for vision-returned
pixel coordinates (shadow DOM / custom web component targets not in a11y tree).
"""

from __future__ import annotations

from agent_first_browse.workers.decision import (
    _bounded_prompt_section,
    _prompt_char_limit,
    _worker_prompt_limit,
    invoke_worker,
)
from agent_first_browse.workers.deterministic import (
    _remove_human_assistance_action,
    _survey_fast_path,
)
from agent_first_browse.workers.prompt_builder import (
    build_system_prompt,
    survey_focus_instructions,
)
from agent_first_browse.workers.schemas import QueuedPageAction, WorkerAction
