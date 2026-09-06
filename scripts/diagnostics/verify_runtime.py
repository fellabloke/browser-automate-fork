"""Deterministic checks for the packaged runtime contracts."""

import os

from agent_first_browse.agent import graph as brain_graph
from agent_first_browse.agent import routing as moe_router
from agent_first_browse.cognition import core as cognition
from agent_first_browse.cognition.prm import ChecklistItem
from agent_first_browse.promotion.browser_promoter.browser_warmup import extract_target_url_from_objective
from agent_first_browse.workers.base import WorkerAction

print("=== Verification ===")
print("All modules imported successfully!")
fields = list(WorkerAction.model_fields.keys())
print(f"WorkerAction fields: {fields}")
assert "ask_user" not in WorkerAction.model_fields["action_type"].description
assert "missing_data" not in fields
ci = ChecklistItem(id=0, description="test", weight=2.0)
assert ci.score == 0.0
ci.status = "done"
assert ci.score == 2.0
ci2 = ChecklistItem(id=1, description="nav step", weight=0.5)
ci2.status = "done"
assert ci2.score == 0.5
expected_prm_cadence = max(1, int(os.getenv("PRM_AUDIT_EVERY", "4")))
assert cognition.PRM_AUDIT_EVERY == expected_prm_cadence
assert extract_target_url_from_objective("Navigate to Amazon (https://www.amazon.in).") == "https://www.amazon.in"
assert extract_target_url_from_objective("Go to https://github.com/user/repo.") == "https://github.com/user/repo"
print("ALL VERIFICATION PASSED")
