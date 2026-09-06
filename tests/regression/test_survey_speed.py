"""Speed-path regression tests: semantic progress, transactions and recipes."""

from __future__ import annotations

from agent_first_browse.survey.benchmarks import SurveyBenchmarkMetrics
from agent_first_browse.survey.context import (
    canonical_survey_url,
    prepare_survey_transaction,
    sparse_survey_dom,
    survey_action_attempt_key,
    survey_failure_fingerprint,
    survey_gate_violation,
    survey_ineffective_action_violation,
    survey_interaction_fingerprint,
    survey_page_fingerprint,
    survey_perception_wait_mode,
    trusted_consent_action,
    unsupported_survey_requirement,
)
from agent_first_browse.survey.recipes import SurveyRecipeMemory, survey_page_recipe_signature
from agent_first_browse.workers.base import _survey_fast_path


def _button(text: str, **extra):
    return {"kind": "button", "text": text, "control_type": "button", **extra}


def _radio(text: str, group: str, selected: bool = False):
    return {
        "kind": "input", "text": text, "control_type": "radio",
        "choice_group": group, "group_label": group, "selected": selected,
    }


def test_question_like_consent_is_a_deterministic_navigation_action():
    selector_map = {
        "e1": _button("Agree and Continue", visible=True, disabled=False),
    }
    action = trusted_consent_action(
        selector_map,
        "We would like your consent to collect demographics. By clicking Agree and Continue "
        "you agree to the terms and privacy policy.",
    )
    assert action == {
        "verb": "click", "element_id": "e1", "answer_basis": "page_navigation",
        "proposal_source": "deterministic_consent", "vision_requested": False,
        "expected_change": "Consent control disappears or the survey page advances.",
    }
    assert survey_gate_violation(
        action, selector_map,
        page_text="We would like your consent to collect demographics. By clicking Agree and Continue "
        "you agree to the terms and privacy policy.",
    ) == ""


def test_consent_classifier_rejects_generic_and_answer_controls():
    assert trusted_consent_action(
        {"e1": _button("Continue")}, "Do you agree to answer this question?"
    ) is None


def test_consent_fast_path_is_model_free_and_snapshot_bound():
    action = _survey_fast_path({
        "continuous_survey_mode": True,
        "snapshot_revision": "rev-consent-1",
        "page_text": "Consent to collect information. Agree and Continue to proceed.",
        "selector_map": {"e1": _button("Agree and Continue")},
    }, set())
    assert action["verb"] == "click"
    assert action["element_id"] == "e1"
    assert action["proposal_source"] == "deterministic_consent"
    assert trusted_consent_action(
        {"e1": {**_radio("I agree", "consent"), "visible": True}},
        "Please select your answer.",
    ) is None
    assert trusted_consent_action(
        {"e1": _button("Agree and Continue", disabled=True)}, "Consent"
    ) is None
    assert trusted_consent_action(
        {"e1": _button("Agree and Continue"), "e2": _button("Accept and Continue")},
        "Consent is required to proceed.",
    ) is None


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


def test_sparse_question_is_detected_when_only_next_survives_ranking():
    assert sparse_survey_dom(
        "What is your age? Please select one answer.",
        {"e1": _button("Next")},
    ) is True


def test_sparse_question_does_not_allow_next_as_a_probe():
    reason = survey_gate_violation(
        {"verb": "click", "element_id": "e1"},
        {"e1": _button("Next")},
        page_text="What is your age? Please select one answer.",
    )
    assert "native answer controls" in reason


def test_sparse_question_uses_bounded_wait_without_model_or_vision():
    state = {
        "continuous_survey_mode": True,
        "snapshot_revision": "sparse-r1",
        "page_text": "What is your age? Please select one answer.",
        "selector_map": {"e1": _button("Next")},
    }
    first = _survey_fast_path(state, set())
    assert first["verb"] == "wait"
    assert first["gate_reason_code"] == "SURVEY_NATIVE_CONTROLS_MISSING"
    second = _survey_fast_path({**state, "survey_hold_identity": first["held_action_identity"],
                                "survey_hold_count": first["hold_count"]}, set())
    assert second["hold_count"] == 2
    assert second["verb"] == "wait"


def test_recovered_native_control_is_not_sparse():
    assert sparse_survey_dom(
        "What is your age?",
        {"s1": _radio("25-34", "age")},
    ) is False


def test_sparse_failure_fingerprint_ignores_query_values():
    first = survey_failure_fingerprint("https://survey.test/q?token=one", kind="sparse_dom")
    second = survey_failure_fingerprint("https://survey.test/q?token=two", kind="sparse_dom")
    assert first == second


def test_benchmark_tracks_duplicate_and_unnecessary_actions():
    metrics = SurveyBenchmarkMetrics()
    metrics.record({"kind": "vision_cache_bypass"})
    metrics.record({"kind": "duplicate_action"})
    metrics.record({"kind": "unnecessary_action"})
    metrics.record({"kind": "valid_action", "elapsed_ms": 125.0})
    assert metrics.summary() == {
        "provider_attempts": 0, "model_attempts": 0, "vision_calls": 0,
        "vision_cache_hits": 0, "vision_cache_bypasses": 1,
        "duplicate_actions": 1, "same_state_actions": 0, "captcha_attempts": 0,
        "unnecessary_actions": 1, "valid_actions": 1, "final_success": False,
        "total_elapsed_ms": 0.0, "first_valid_action_ms": 125.0,
    }


def test_unsupported_media_gate_only_matches_produced_responses():
    assert unsupported_survey_requirement("Please record a video response explaining your choice") == "video_response"
    assert unsupported_survey_requirement("Record your voice answer using the microphone") == "audio_response"
    assert unsupported_survey_requirement("Listen to this audio clip and choose what you hear") == ""
    assert unsupported_survey_requirement("Watch this video before answering") == ""


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


def test_paidwork_loading_budget_recovers_instead_of_repeating_wait():
    action = _survey_fast_path({
        "continuous_survey_mode": True,
        "current_url": "https://www.paidwork.com/earn/filling-out",
        "page_text": "Earn Fill out Profile",
        "selector_map": {"e1": _button("Earn")},
        "paidwork_selection_waits": 3,
    }, set())
    assert action and action["verb"] == "goto"
    assert action["url"] == "https://www.paidwork.com/earn"


def test_paidwork_nested_provider_route_does_not_replay_recipe_or_offer_nav():
    action = _survey_fast_path({
        "continuous_survey_mode": True,
        "current_url": "https://www.paidwork.com/earn/filling-out/bitlabs",
        "page_text": "Earn Fill out Profile",
        "selector_map": {"e7": _button("Earn")},
    }, set())
    assert action is None


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
