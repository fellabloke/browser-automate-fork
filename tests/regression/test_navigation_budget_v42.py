"""Regression tests for bounded survey dashboard navigation and failover."""

import asyncio

from agent_first_browse.actions import tools as mcp_tools
from agent_first_browse.verification import overwatch
from agent_first_browse.verification.overwatch import _action_execution_confirmed
from agent_first_browse.survey.context import (
    compact_survey_url,
    is_verified_survey_page_transition,
    recently_failed_survey_offer_ids,
    survey_completion_evidence,
    survey_cycle_cleanup_updates,
    survey_dashboard_stall_step_limit,
    survey_gate_violation,
    survey_nonresponse_violation,
    survey_page_fingerprint,
)
from agent_first_browse.survey.site_quirks import (
    fresh_dashboard_after_boundary,
    fresh_dashboard_after_completion,
    qmee_active_survey_action,
)
from agent_first_browse.workers.base import (
    WorkerAction,
    _bounded_prompt_section,
    _remove_human_assistance_action,
    _survey_fast_path,
)


def test_dashboard_stall_limit_is_short_and_configurable(monkeypatch):
    monkeypatch.setenv("SURVEY_DASHBOARD_STALL_STEPS", "7")
    assert survey_dashboard_stall_step_limit() == 7
    monkeypatch.setenv("SURVEY_DASHBOARD_STALL_STEPS", "1")
    assert survey_dashboard_stall_step_limit() == 3


def test_cycle_cleanup_resets_dashboard_stall_counter():
    updates = survey_cycle_cleanup_updates(
        {"step_number": 20, "survey_dashboard_stall_steps": 8},
        outcome="abandoned:unchanged_page_timeout",
    )
    assert updates["survey_dashboard_stall_steps"] == 0
    assert updates["survey_last_boundary_outcome"] == "abandoned:unchanged_page_timeout"


def test_completion_requires_terminal_outcome_not_intro_or_terms_copy():
    assert survey_completion_evidence(
        "Welcome! Thank you for taking part in this survey. Click '>' to begin."
    ) == ""
    assert survey_completion_evidence(
        "Terms and conditions for a new survey. Complete every question carefully. Thank you."
    ) == ""
    assert survey_completion_evidence(
        "These are all the questions we have for you today. Thank you for your time. "
        "Click Submit to finish."
    ) == ""
    assert survey_completion_evidence(
        "Thank you for completing the survey. Your responses have been recorded."
    )
    assert survey_completion_evidence("Your reward has been credited to your account.")


def test_disabled_and_exit_controls_are_never_clicked_as_forward_progress():
    assert "disabled" in survey_gate_violation(
        {"verb": "click", "element_id": "e1"},
        {"e1": {"text": "Next east [disabled]", "kind": "button"}},
    ).lower()
    assert "exits or closes" in survey_gate_violation(
        {"verb": "click", "element_id": "e2"},
        {"e2": {"text": "westExit", "kind": "button"}},
    )
    assert survey_gate_violation(
        {"verb": "click", "element_id": "e3"},
        {"e3": {
            "text": "I would exit the situation", "kind": "div",
            "control_type": "radio", "selected": False,
        }},
    ) == ""


def test_failed_best_offer_is_quarantined_and_next_best_becomes_eligible():
    selector_map = {
        "e1": {"text": "£1.00 • 10 minutes", "kind": "div"},
        "e2": {"text": "£0.80 • 10 minutes", "kind": "div"},
    }
    failed = recently_failed_survey_offer_ids(
        selector_map,
        [{
            "verb": "click", "element_id": "e1",
            "target_name": "£1.00 • 10 minutes",
            "outcome": "→ CLICK INEFFECTIVE: all strategies exhausted",
            "pre_url": "https://dashboard.test/surveys",
        }],
        current_url="https://dashboard.test/surveys",
    )
    assert failed == {"e1"}
    assert "just attempted" in survey_gate_violation(
        {"verb": "click", "element_id": "e1"}, selector_map,
        unavailable_offer_ids=failed,
    )
    assert survey_gate_violation(
        {"verb": "click", "element_id": "e2"}, selector_map,
        unavailable_offer_ids=failed,
    ) == ""


def test_executor_failure_and_unverified_dispatch_cannot_become_dom_progress():
    assert _action_execution_confirmed("→ OK (DOM changed)") is True
    assert _action_execution_confirmed("→ DRAG FAILED: target missed") is False
    assert _action_execution_confirmed(
        "→ DISPATCHED ONCE [verification pending]: no observable effect"
    ) is False
    assert is_verified_survey_page_transition("old", "new", "→ OK (DOM changed)")
    assert not is_verified_survey_page_transition(
        "old", "new", "→ CLICK INEFFECTIVE: all strategies exhausted"
    )
    assert not is_verified_survey_page_transition(
        "old", "new", "→ OK (DOM changed)", action={"verb": "type"}
    )


def test_autocomplete_suggestions_do_not_change_question_identity():
    base = {
        "e1": {"control_type": "text", "question_key": "occupation|What is your occupation?"},
    }
    with_suggestions = {
        **base,
        "e2": {"control_type": "option", "question_key": "occupation|What is your occupation?"},
        "e3": {"control_type": "option", "question_key": "occupation|What is your occupation?"},
    }
    assert survey_page_fingerprint("Occupation", base) == survey_page_fingerprint(
        "Occupation\nSocial care professionals\nOther", with_suggestions
    )


def test_active_survey_never_accepts_skip_or_empty_answer():
    assert "Never skip" in survey_nonresponse_violation(
        {"verb": "skip_question"}, {}
    )
    assert "empty" in survey_nonresponse_violation(
        {"verb": "type", "element_id": "e1", "text": ""}, {"e1": {}}
    )
    assert "empty" in survey_nonresponse_violation(
        {"verb": "select_option", "element_id": "e1", "text": ""}, {"e1": {}}
    )
    assert "non-response" in survey_nonresponse_violation(
        {"verb": "click", "element_id": "e1"},
        {"e1": {"text": "Prefer not to say", "control_type": "radio"}},
    )


def test_prompt_compaction_keeps_boundaries_and_obeys_exact_cap():
    source = "HEADER\n" + ("middle\n" * 1000) + "LATEST CONTROL"
    bounded = _bounded_prompt_section(source, 500)
    recent = _bounded_prompt_section(source, 300, recent=True)
    assert len(bounded) <= 500
    assert bounded.startswith("HEADER")
    assert bounded.endswith("LATEST CONTROL")
    assert len(recent) <= 300
    assert recent.endswith("LATEST CONTROL")


def test_long_screener_url_keeps_route_but_not_demographic_values():
    url = (
        "https://screener.test/start?survey_id=52050068&postcode=KY8&age=20&"
        + "&".join(f"profile_{index}=sensitive-{index}" for index in range(80))
    )
    compact = compact_survey_url(url, max_chars=240)
    assert compact.startswith("https://screener.test/start?survey_id=…")
    assert "postcode=…" in compact
    assert "KY8" not in compact and "sensitive" not in compact
    assert len(compact) <= 240


def test_qmee_completion_quirk_requests_a_fresh_dashboard_tab():
    assert fresh_dashboard_after_completion(
        "https://www.qmee.com/en-gb/surveys?message=completed"
    ) == "https://www.qmee.com/en-gb/surveys"
    assert fresh_dashboard_after_completion(
        "https://surveys.gobranded.com/members"
    ) == "https://surveys.gobranded.com/members"


def test_qmee_any_boundary_recreates_dashboard():
    assert fresh_dashboard_after_boundary(
        "https://www.qmee.com/en-gb/surveys",
        "abandoned:unchanged_page_timeout",
    ) == "https://www.qmee.com/en-gb/surveys"
    assert fresh_dashboard_after_boundary(
        "https://surveys.gobranded.com/members", "abandoned:load_failed"
    ) == ""


def test_qmee_abandonment_executes_with_fresh_dashboard(monkeypatch):
    captured = {}

    async def abandon(url, *, fresh_dashboard=False):
        captured.update(url=url, fresh_dashboard=fresh_dashboard)
        return {"success": True, "url": url, "error": ""}

    monkeypatch.setattr(mcp_tools, "mcp_abandon_survey", abandon)
    result = asyncio.run(overwatch._execute_action({
        "verb": "abandon_survey",
        "url": "https://www.qmee.com/en-gb/surveys",
        "survey_boundary_reason": "abandoned:unchanged_page_timeout",
    }, None))

    assert result.startswith("→ OK")
    assert captured["fresh_dashboard"] is True


def test_qmee_active_marker_uses_truthful_previous_boundary():
    selector_map = {
        "e1": {"kind": "button", "text": "I'd like to do this survey instead"},
        "e5": {"kind": "link", "text": "I finished it already 🏃‍♀️"},
        "e6": {"kind": "link", "text": "It was broken/stuck 💔"},
    }
    element_id, reason = qmee_active_survey_action(
        "https://router.qmee.com/q-feedback.html",
        "You are already doing a survey!? Sure. What was the problem with the other one?",
        selector_map,
        previous_boundary="abandoned:unchanged_page_timeout",
    )
    assert (element_id, reason) == ("e6", "broken_or_stuck")

    action = _survey_fast_path({
        "continuous_survey_mode": True,
        "current_url": "https://router.qmee.com/q-feedback.html",
        "page_text": "You are already doing a survey!? What was the problem with the other one?",
        "selector_map": selector_map,
        "survey_last_boundary_outcome": "abandoned:unchanged_page_timeout",
    }, set())
    assert action["element_id"] == "e6"
    assert action["answer_basis"] == "page_navigation"


def test_qmee_active_marker_first_conflict_stage_uses_new_survey_button():
    selector_map = {
        "e1": {"kind": "button", "text": "I'd like to do this survey instead 👎"},
    }
    element_id, stage = qmee_active_survey_action(
        "https://router.qmee.com/q-feedback.html",
        "You are already doing a survey!? If you'd prefer to continue with this survey instead, click below.",
        selector_map,
        previous_boundary="abandoned:unchanged_page_timeout",
    )
    assert (element_id, stage) == ("e1", "use_new_survey")


def test_legacy_human_help_action_is_fully_rewritten():
    decision = WorkerAction(
        screen_state="A drag target is unavailable.",
        previous_action_result="No progress.",
        goal_progress="Survey active.",
        question_text="Drag 35 into the square.",
        answer_basis="unknown_needs_vision",
        reasoning="Ask the person to complete it.",
        expected_change="The user will drag it.",
        action_type="ask_user",
    )
    rewritten = _remove_human_assistance_action(decision)
    assert rewritten.action_type == "wait"
    assert rewritten.needs_vision is True
    assert "user" not in rewritten.reasoning.lower()
    assert "user" not in rewritten.expected_change.lower()
