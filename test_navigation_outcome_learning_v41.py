"""Regression coverage for ambiguous-label A↔B navigation loops."""

from __future__ import annotations

import asyncio

from overwatch import overwatch_node
from stagnation import (
    detect_navigation_cycle,
    detect_stagnation,
    navigation_cycle_action_violation,
)


def _paidwork_cycle_history():
    return [
        {
            "verb": "click", "element_id": "e9", "target_name": "Fill out surveys",
            "pre_url": "https://www.paidwork.com/earn/filling-out/bitlabs",
            "url": "https://www.paidwork.com/earn/filling-out",
            "pre_page_fingerprint": "a", "post_page_fingerprint": "b",
        },
        {
            "verb": "click", "element_id": "e1", "target_name": "Fill out",
            "target_context": "/earn/filling-out/bitlabs · BitLabs",
            "pre_url": "https://www.paidwork.com/earn/filling-out",
            "url": "https://www.paidwork.com/earn/filling-out/bitlabs",
            "pre_page_fingerprint": "b", "post_page_fingerprint": "a",
        },
        {
            "verb": "click", "element_id": "e9", "target_name": "Fill out surveys",
            "pre_url": "https://www.paidwork.com/earn/filling-out/bitlabs",
            "url": "https://www.paidwork.com/earn/filling-out",
            "pre_page_fingerprint": "a", "post_page_fingerprint": "b",
        },
    ]


def test_navigation_cycle_learns_the_loop_closing_element():
    signal = detect_navigation_cycle(
        _paidwork_cycle_history(),
        current_url="https://www.paidwork.com/earn/filling-out",
        current_fingerprint="fresh-dynamic-fingerprint",
    )

    assert signal.detected is True
    assert signal.blocked_action["element_id"] == "e1"
    assert signal.blocked_action["target_name"] == "Fill out"
    assert "ambiguous" in signal.note.lower()


def test_same_label_sibling_remains_available():
    signal = detect_navigation_cycle(
        _paidwork_cycle_history(),
        current_url="https://www.paidwork.com/earn/filling-out",
    )
    state = {
        "navigation_cycle_note": signal.note,
        "navigation_cycle_blocked_action": signal.blocked_action,
        "selector_map": {
            "e1": {"hint": "/earn/filling-out/bitlabs · BitLabs", "x": 708, "y": 441},
            "e3": {"hint": "/earn/filling-out/prime · Prime Surveys"},
            "e7": {"hint": "/earn/filling-out/bitlabs · BitLabs"},
        },
    }

    assert navigation_cycle_action_violation(
        state, {"verb": "click", "element_id": "e1", "target_name": "Fill out"}
    )
    assert not navigation_cycle_action_violation(
        state, {"verb": "click", "element_id": "e3", "target_name": "Fill out"}
    )
    assert navigation_cycle_action_violation(
        state, {"verb": "click", "element_id": "e7", "target_name": "Fill out"}
    )
    assert navigation_cycle_action_violation(
        state, {"verb": "click", "element_id": None, "x": 710, "y": 442}
    )


def test_action_cycle_is_stagnation_even_when_each_leg_looks_like_progress():
    signal = detect_stagnation({
        "same_url_streak": 0,
        "goal_score_window": [],
        "loop_signatures": ["A", "B", "A", "B"],
        "recent_survey_progress": True,
    })

    assert signal.stuck is True
    assert "short action cycle" in signal.note


def test_overwatch_blocks_learned_cycle_action_before_execution():
    signal = detect_navigation_cycle(
        _paidwork_cycle_history(),
        current_url="https://www.paidwork.com/earn/filling-out",
    )
    updates = asyncio.run(overwatch_node(
        {
            "proposed_action": {
                "verb": "click", "element_id": "e1", "target_name": "Fill out",
            },
            "navigation_cycle_note": signal.note,
            "navigation_cycle_blocked_action": signal.blocked_action,
        },
        page=None,
        critic=None,
    ))

    assert updates["overwatch_verdict"] == "retry"
    assert updates["action_outcome"] == "NAVIGATION_CYCLE_BLOCKED"
    assert updates["proposed_action"] is None


def test_overwatch_does_not_commit_a_model_timeout_as_progress():
    updates = asyncio.run(overwatch_node(
        {"proposed_action": None, "step_number": 17},
        page=None,
        critic=None,
    ))

    assert updates["overwatch_verdict"] == "retry"
    assert "step_number" not in updates
    assert "no executable action" in updates["correction_context"]
