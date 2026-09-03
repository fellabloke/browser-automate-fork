"""Speed-path regression tests: semantic progress, transactions and recipes."""

from __future__ import annotations

from survey_context import (
    canonical_survey_url,
    prepare_survey_transaction,
    survey_action_attempt_key,
    survey_ineffective_action_violation,
    survey_interaction_fingerprint,
    survey_page_fingerprint,
    survey_perception_wait_mode,
)
from survey_recipe_memory import SurveyRecipeMemory, survey_page_recipe_signature
from workers.base_worker import _survey_fast_path


def _button(text: str, **extra):
    return {"kind": "button", "text": text, "control_type": "button", **extra}


def _radio(text: str, group: str, selected: bool = False):
    return {
        "kind": "input", "text": text, "control_type": "radio",
        "choice_group": group, "group_label": group, "selected": selected,
    }


def test_canonical_url_ignores_values_but_keeps_route_keys():
    first = canonical_survey_url("HTTPS://Survey.Test/q?token=one&pid=7#top")
    second = canonical_survey_url("https://survey.test/q?pid=99&token=two#other")
    assert first == second == "https://survey.test/q?pid&token"
    assert canonical_survey_url("https://survey.test/other?pid=99") != second


def test_wait_mode_reserves_long_wait_for_real_navigation():
    base = {
        "objective": "Complete surveys continuously",
        "current_url": "https://survey.test/q",
        "action_outcome": "→ OK (control state verified)",
        "page_fsm": "READY",
    }
    assert survey_perception_wait_mode(base, "https://survey.test/q") == "same_page"
    assert survey_perception_wait_mode(base, "https://survey.test/q2") == "navigation"
    assert survey_perception_wait_mode(
        {**base, "action_outcome": "→ OK (navigated)"}, "https://survey.test/q"
    ) == "navigation"


def test_interaction_fingerprint_notices_answer_state_not_coordinates():
    before = {"e1": {**_radio("Yes", "age"), "x": 10, "y": 20}}
    moved = {"e1": {**_radio("Yes", "age"), "x": 100, "y": 200}}
    answered = {"e1": {**_radio("Yes", "age", selected=True), "x": 100, "y": 200}}
    assert survey_interaction_fingerprint(before) == survey_interaction_fingerprint(moved)
    assert survey_interaction_fingerprint(before) != survey_interaction_fingerprint(answered)


def test_simple_answer_gets_guarded_automatic_next():
    selector_map = {
        "e1": _radio("Yes", "q1"),
        "e2": _radio("No", "q1"),
        "e3": _button("Next"),
    }
    queue, reason = prepare_survey_transaction(
        {"verb": "click", "element_id": "e1", "answer_basis": "configured_profile_fact"},
        [], selector_map, page_text="Are you employed?", continuous_mode=True,
    )
    assert reason == ""
    assert [(item["verb"], item["element_id"]) for item in queue] == [("click", "e3")]


def test_matrix_transaction_finishes_all_rows_before_next():
    selector_map = {
        "e1": _radio("Row one Yes", "row1"),
        "e2": _radio("Row one No", "row1"),
        "e3": _radio("Row two Yes", "row2"),
        "e4": _radio("Row two No", "row2"),
        "e5": _button("Continue"),
    }
    page_text = "Please answer every row."
    incomplete, _ = prepare_survey_transaction(
        {"verb": "click", "element_id": "e1"}, [], selector_map,
        page_text=page_text, continuous_mode=True,
    )
    assert incomplete == []

    complete, reason = prepare_survey_transaction(
        {"verb": "click", "element_id": "e1"},
        [
            {"verb": "click", "element_id": "e3"},
            {"verb": "click", "element_id": "e5"},
        ],
        selector_map, page_text=page_text, continuous_mode=True,
    )
    assert reason == ""
    assert [item["element_id"] for item in complete] == ["e3", "e5"]


def test_forward_action_cannot_be_mid_transaction():
    selector_map = {"e1": _radio("Yes", "q"), "e2": _button("Next"), "e3": _button("Help")}
    queue, reason = prepare_survey_transaction(
        {"verb": "click", "element_id": "e1"},
        [{"verb": "click", "element_id": "e2"}, {"verb": "click", "element_id": "e3"}],
        selector_map, page_text="Choose one", continuous_mode=True,
    )
    assert queue == []
    assert "final" in reason


def test_no_effect_counter_quarantines_exact_action():
    state = {
        "current_url": "https://survey.test/q?token=one",
        "survey_page_fingerprint": survey_page_fingerprint("Question one"),
        "survey_interaction_fingerprint": "form",
        "history": [],
    }
    action = {"verb": "click", "element_id": "e1", "target_name": "Fill out surveys"}
    key = survey_action_attempt_key(state, action)
    state["survey_action_no_effect_counts"] = {key: 2}
    assert "quarantined" in survey_ineffective_action_violation(state, action)


def test_dashboard_fast_path_uses_best_reward_per_minute():
    state = {
        "continuous_survey_mode": True,
        "current_url": "https://panel.test/dashboard",
        "page_text": "Available surveys",
        "selector_map": {
            "e1": _button("£1.00 · 20 minutes"),
            "e2": _button("£0.80 · 8 minutes"),
        },
    }
    action = _survey_fast_path(state, set())
    assert action and action["element_id"] == "e2"
    assert action["answer_basis"] == "reward_per_minute"


def test_recipe_requires_two_successes_and_retires_after_failure(tmp_path):
    memory = SurveyRecipeMemory(tmp_path / "recipes.db")
    url = "https://survey.test/repeated?token=one"
    page_text = "Welcome. Click Next to begin."
    selector_map = {"e7": _button("Next")}
    action = {
        "verb": "click", "element_id": "e7", "target_name": "Next",
        "answer_basis": "page_navigation", "question_text": page_text,
    }
    assert memory.recall(
        url=url, page_text=page_text, selector_map=selector_map, profile={}
    ) is None
    memory.observe_success(
        url=url, page_text=page_text, selector_map=selector_map, action=action,
        verified_transition=True,
    )
    assert memory.recall(
        url=url, page_text=page_text, selector_map=selector_map, profile={}
    ) is None
    memory.observe_success(
        url=url, page_text=page_text, selector_map=selector_map, action=action,
        verified_transition=True,
    )
    recalled = memory.recall(
        url="https://survey.test/repeated?token=changed",
        page_text=page_text, selector_map=selector_map, profile={},
    )
    assert recalled and recalled["element_id"] == "e7"
    signature = survey_page_recipe_signature(url, page_text, selector_map)
    memory.record_replay_failure(signature, recalled["recipe_action_key"])
    assert memory.recall(
        url=url, page_text=page_text, selector_map=selector_map, profile={}
    ) is None
