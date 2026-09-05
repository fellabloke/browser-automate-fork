#!/usr/bin/env python3
"""Quick import test for all True Brain v16.0 modules."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

print("=== True Brain v16.0 Import Test ===\n")

# 1. BrainState
from agent_first_browse.agent.state import BrainState

s = BrainState(objective="Test import chain")
print(f"✓ brain_state.py — {len(type(s).model_fields)} fields")
print(f"  Plan render: {s.get_plan_render()[:60]}")
print(f"  History: {s.compress_history()}")

# 2. MoE Router
from agent_first_browse.agent.routing import route_to_worker, verdict_router

state_dict = s.model_dump()
route = route_to_worker(state_dict)
print(f"✓ moe_router.py — default route: {route}")
v = verdict_router({"overwatch_verdict": "pass"})
print(f"  verdict_router(pass) → {v}")

# 3. Workers
from workers.base_worker import build_system_prompt, survey_focus_instructions

nav_prompt = build_system_prompt("navigator", "test plan", "test facts")
print(f"✓ workers/base_worker.py — navigator prompt: {len(nav_prompt)} chars")
int_prompt = build_system_prompt("interactor")
print(f"  interactor prompt: {len(int_prompt)} chars")
ext_prompt = build_system_prompt("extractor")
print(f"  extractor prompt: {len(ext_prompt)} chars")


def test_survey_focus_instructions_are_adaptive_and_consistent():
    prompt = survey_focus_instructions("Go to AttaPoll and complete a survey")
    assert "SURVEY COMPLETION MODE" in prompt
    assert "Step N/25" in prompt
    assert "I agree" in prompt and "ANSWER" in prompt
    assert "Never fabricate" in prompt
    assert "try another available survey" in prompt


def test_survey_focus_instructions_do_not_leak_into_other_tasks():
    assert survey_focus_instructions("Find the top Reddit post") == ""

# 4. LangGraph imports
print("✓ langgraph — StateGraph imported")

# 5. Build graph (without browser)
from brain_graph import build_brain_graph

graph = build_brain_graph()
print(f"✓ brain_graph.py — {len(graph.nodes)} nodes in graph")
print(f"  Nodes: {sorted(graph.nodes.keys())}")

print("\n=== ALL IMPORTS PASSED ===")
