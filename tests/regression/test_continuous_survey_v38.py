"""Regression tests for continuous survey execution and SPA progress tracking."""

from __future__ import annotations

import asyncio

from agent_first_browse.agent import graph as brain_graph
from agent_first_browse.agent.state import BrainState
from agent_first_browse.agent.routing import route_to_worker
from agent_first_browse.verification.overwatch import _action_loop_signature, _layer_4_cove_check
from agent_first_browse.cognition.stagnation import detect_stagnation
from agent_first_browse.survey.context import (
    blocking_popup_action_id,
    build_survey_handoff,
    captcha_field_state,
    has_recent_survey_progress,
    is_continuous_survey_mission,
    rank_survey_offers,
    reconcile_captcha_vision,
    rolling_continuous_budget,
    survey_completion_evidence,
    survey_failure_kind,
    survey_gate_violation,
    survey_page_fingerprint,
)
from agent_first_browse.perception.vision import VisionVerdict


def test_survey_failure_kind_detects_provider_exit_states():
    assert survey_failure_kind("Sorry, you have been disqualified from this survey") == "disqualified"
    assert survey_failure_kind("Unfortunately, you haven't qualified for any surveys at this time.") == "disqualified"
    assert survey_failure_kind("This survey is full") == "quota_full"
    assert survey_failure_kind("The survey failed to load") == "load_failed"
    assert survey_failure_kind("Question 4 of 20") == ""
    assert survey_failure_kind(
        "Congratulations, you have qualified - be open & honest. If disqualified, return later."
    ) == ""


def test_survey_gate_blocks_informational_provider_links():
    violation = survey_gate_violation(
        {"verb": "click", "element_id": "e2"},
        {"e2": {
            "kind": "link", "role": "link", "text": "Kantar data controllers",
            "href": "https://www.kantar.com/Surveys-data-controllers",
        }},
        page_text="Survey consent",
        continuous_mode=True,
    )
    assert "informational/legal/provider link" in violation


def test_survey_missions_are_continuous_by_default_with_explicit_one_shot_override(monkeypatch):
    monkeypatch.delenv("SURVEY_CONTINUOUS_MODE", raising=False)
    assert is_continuous_survey_mission("Complete the best available surveys")
    assert is_continuous_survey_mission("Complete a survey")
    assert not is_continuous_survey_mission("Complete exactly one survey")
    assert not is_continuous_survey_mission("Buy an item")

    monkeypatch.setenv("SURVEY_CONTINUOUS_MODE", "false")
    assert not is_continuous_survey_mission("Complete surveys")


def test_question_fingerprint_tracks_spa_progress_without_a_url_change():
    first = survey_page_fingerprint("Question 3: Which animals do you own? 04:52")
    same = survey_page_fingerprint("  Question 3: Which animals do you own? 04:51  ")
    following = survey_page_fingerprint("Question 4: Which shops have you visited? 04:50")

    assert first == same
    assert first != following


def test_recent_question_progress_suppresses_false_stagnation():
    stuck = detect_stagnation({
        "same_url_streak": 9,
        "goal_score_window": [0.5, 0.5, 0.5, 0.5],
        "loop_signatures": [],
    })
    moving = detect_stagnation({
        "same_url_streak": 9,
        "goal_score_window": [0.5, 0.5, 0.5, 0.5],
        "loop_signatures": [],
        "recent_survey_progress": True,
    })

    assert stuck.stuck
    assert not moving.stuck and moving.level == 0
    assert has_recent_survey_progress({"survey_page_advanced": True})
    assert not has_recent_survey_progress({
        "history": [{"target_name": "Next", "outcome": "→ OK (structure changed)"}],
    })
    assert has_recent_survey_progress({
        "history": [{"survey_transition_verified": True}],
    })


def test_same_next_button_on_different_questions_is_not_the_same_loop():
    base = {"current_url": "https://provider.test/survey"}
    first = _action_loop_signature(
        {**base, "survey_page_fingerprint": "question-one"}, "click", "e18"
    )
    second = _action_loop_signature(
        {**base, "survey_page_fingerprint": "question-two"}, "click", "e18"
    )

    assert first != second


def test_continuous_budget_rolls_and_router_does_not_finalize_at_old_limit():
    assert rolling_continuous_budget(20, 25) == 25
    assert rolling_continuous_budget(21, 25) == 50
    assert route_to_worker({
        "step_number": 25,
        "max_steps": 25,
        "continuous_survey_mode": True,
        "plan_steps": [],
    }) != "finalize"


def test_continuous_mode_rejects_done_and_explains_next_cycle():
    violation = survey_gate_violation(
        {"verb": "done"}, {}, continuous_mode=True
    )
    assert "continuous survey session" in violation.lower()

    handoff = build_survey_handoff({
        "objective": "Complete surveys",
        "continuous_survey_mode": True,
        "survey_cycles_completed": 2,
    })
    assert "CONTINUOUS SURVEY MODE: ACTIVE" in handoff
    assert "SURVEY CYCLES COMPLETED THIS RUN: 2" in handoff
    assert "return to the dashboard" in handoff


def test_qualification_is_not_completion_evidence():
    assert not survey_completion_evidence(
        "Congratulations, you have qualified. Click Start Survey to begin."
    )
    assert survey_completion_evidence(
        "Thank you. You have successfully completed this survey."
    )


def test_overwatch_blocks_terminal_done_without_calling_the_outcome_model():
    updates = asyncio.run(_layer_4_cove_check(
        {
            "continuous_survey_mode": True,
            "step_number": 12,
            "objective": "Complete surveys",
        },
        page=None,
        updates={},
    ))

    assert updates["overwatch_verdict"] == "retry"
    assert updates["mission_success"] is False
    assert "paid survey" in updates["correction_context"].lower()


def test_commit_rolls_budget_and_clears_an_accidental_success_latch(monkeypatch):
    monkeypatch.setattr(brain_graph, "_PRM_CRITIC", None)
    state = BrainState(
        objective="Complete surveys",
        continuous_survey_mode=True,
        step_number=21,
        max_steps=25,
        mission_success=True,
        done_evidence="qualification page",
        proposed_action={"verb": "click", "target_name": "Start Survey"},
        action_outcome="→ OK (navigated)",
    )

    updates = asyncio.run(brain_graph.commit_node(state))

    assert updates["step_number"] == 22
    assert updates["max_steps"] == 50
    assert updates["mission_success"] is False
    assert updates["done_evidence"] == ""


def test_points_offers_are_ranked_by_points_per_minute():
    offers = rank_survey_offers({
        "e1": {"text": "88 points / 9 minutes", "kind": "button"},
        "e2": {"text": "105 pts / 29 mins", "kind": "button"},
        "e3": {"text": "16 points / 11 min", "kind": "button"},
    })
    assert [offer.element_id for offer in offers] == ["e1", "e2", "e3"]
    assert offers[0].currency == "points"


def test_popup_first_gate_redirects_a_dashboard_offer_to_close():
    selector_map = {
        "e1": {"text": "£1.20 / 8 minutes", "kind": "button", "in_modal": False},
        "e2": {"text": "Close", "kind": "button", "in_modal": True},
        "e3": {"text": 'input [filled: "answer" 6ch]', "kind": "input", "in_modal": False},
    }
    assert blocking_popup_action_id(selector_map) == "e2"
    violation = survey_gate_violation(
        {"verb": "click", "element_id": "e1"}, selector_map
    )
    assert "blocking popup" in violation.lower() and "[e2]" in violation
    assert survey_gate_violation(
        {"verb": "click", "element_id": "e2"}, selector_map
    ) == ""


def test_qmee_snap_popup_prefers_acknowledge_button_over_dashboard():
    """Qmee's Snap modal has no semantic dialog role; its acknowledgement is still safe."""
    selector_map = {
        "e1": {"text": "£1.20 / 8 minutes", "kind": "button", "in_modal": False},
        "e2": {"text": "Awesome", "kind": "button", "in_modal": True},
    }
    assert blocking_popup_action_id(selector_map) == "e2"

    selector_map["e2"]["text"] = "OK"
    assert blocking_popup_action_id(selector_map) == "e2"


def test_forward_is_blocked_until_every_required_radio_row_is_answered():
    selector_map = {
        "e1": {"text": "Away", "control_type": "radio", "selected": True,
               "choice_group": "name:morning", "group_label": "Morning", "required": True},
        "e2": {"text": "Home", "control_type": "radio", "selected": False,
               "choice_group": "name:morning", "group_label": "Morning", "required": True},
        "e3": {"text": "Away", "control_type": "radio", "selected": False,
               "choice_group": "name:afternoon", "group_label": "Afternoon", "required": True},
        "e4": {"text": "Home", "control_type": "radio", "selected": False,
               "choice_group": "name:afternoon", "group_label": "Afternoon", "required": True},
        "e5": {"text": "Next", "kind": "button"},
    }
    violation = survey_gate_violation(
        {"verb": "click", "element_id": "e5"}, selector_map,
        page_text="Please select one response for each time period.",
    )
    assert "1/2" in violation and "Afternoon" in violation
    repeat = survey_gate_violation(
        {"verb": "click", "element_id": "e1"}, selector_map,
        page_text="Please select one response for each time period.",
    )
    assert "other required rows" in repeat


def test_captcha_filled_code_is_compared_then_submitted_once():
    selector_map = {
        "e1": {"text": 'input [filled: "kxkWfp" 6ch]', "kind": "input"},
        "e2": {"text": "Refresh the image", "kind": "button"},
        "e3": {"text": "Go to next question", "kind": "button"},
    }
    assert captcha_field_state(selector_map) == ("e1", "kxkWfp")
    verdict = VisionVerdict(
        observation="The image reads kxkWfp and the field matches.",
        action_type="type", element_id="e1", text="kxkWfp",
        reasoning="Exact match", confidence=0.97,
    )
    action, resolved, note = reconcile_captcha_vision(
        {"verb": "type", "element_id": "e1"}, verdict, selector_map
    )
    assert resolved and action["verb"] == "click" and action["element_id"] == "e3"
    assert action["captcha_verified"] is True and "matched" in note


def test_uncertain_captcha_refreshes_instead_of_typing_placeholder():
    selector_map = {
        "e1": {"text": "input [empty]", "kind": "input"},
        "e2": {"text": "Refresh the image", "kind": "button"},
        "e3": {"text": "Go to next question [disabled]", "kind": "button", "disabled": True},
    }
    verdict = VisionVerdict(
        observation="Only AA is visible", action_type="type", element_id="e1",
        text="AA", reasoning="uncertain", confidence=0.6,
    )
    action, resolved, _ = reconcile_captcha_vision(
        {"verb": "type", "element_id": "e1"}, verdict, selector_map
    )
    assert resolved and action["verb"] == "click" and action["element_id"] == "e2"
