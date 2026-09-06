import os
import sys
sys.path.insert(0, '.')

from agent_first_browse.agent import graph as brain_graph
from agent_first_browse.cognition import prm as prm_critic
from agent_first_browse.cognition import core as cognition
import agent_first_browse.agent.state as brain_state
from agent_first_browse.agent import routing as moe_router
from agent_first_browse.workers.base import WorkerAction

print("=== V17 Verification ===")
print("All modules imported successfully!")

# Check WorkerAction has new fields
fields = list(WorkerAction.model_fields.keys())
print(f"WorkerAction fields: {fields}")
assert "ask_user" not in WorkerAction.model_fields["action_type"].description
assert "missing_data" not in fields
print("  autonomous action schema (no ask_user): OK")

# Check ChecklistItem weighted scoring
from prm_critic import ChecklistItem
ci = ChecklistItem(id=0, description="test", weight=2.0)
print(f"ChecklistItem pending score (w=2.0): {ci.score}")
assert ci.score == 0.0

ci.status = "done"
print(f"ChecklistItem done score (w=2.0): {ci.score}")
assert ci.score == 2.0, f"Expected 2.0, got {ci.score}"

ci2 = ChecklistItem(id=1, description="nav step", weight=0.5)
ci2.status = "done"
print(f"ChecklistItem done score (w=0.5): {ci2.score}")
assert ci2.score == 0.5

# Check PRM audit frequency
print(f"PRM_AUDIT_EVERY: {cognition.PRM_AUDIT_EVERY}")
expected_prm_cadence = max(1, int(os.getenv("PRM_AUDIT_EVERY", "4")))
assert cognition.PRM_AUDIT_EVERY == expected_prm_cadence, (
    f"Expected {expected_prm_cadence}, got {cognition.PRM_AUDIT_EVERY}"
)

# Check URL extraction fix
from agent_first_browse.promotion.browser_promoter.browser_warmup import extract_target_url_from_objective
url = extract_target_url_from_objective("Navigate to Amazon (https://www.amazon.in).")
print(f"URL extraction test: 'Navigate to Amazon (https://www.amazon.in).' -> '{url}'")
assert url == "https://www.amazon.in", f"Expected 'https://www.amazon.in', got '{url}'"

url2 = extract_target_url_from_objective("Go to https://github.com/user/repo.")
print(f"URL extraction test: 'Go to https://github.com/user/repo.' -> '{url2}'")
assert url2 == "https://github.com/user/repo", f"Expected clean URL, got '{url2}'"

print("\n=== ALL V17 VERIFICATION PASSED ===")
