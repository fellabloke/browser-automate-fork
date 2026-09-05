"""Provider rotation, verified progress, and conservative survey timeout tests."""

from __future__ import annotations

import asyncio
import time

import brain_graph
from brain_state import BrainState
from moe_router import route_to_worker
from survey_context import (
    should_rotate_survey_provider,
    survey_provider_urls,
    survey_semantic_page_identity,
    survey_stuck_watch_updates,
)


def test_provider_urls_are_ordered_and_deduplicated():
    assert survey_provider_urls(
        "https://www.qmee.com/en-gb/surveys;"
        "https://fallback.test/surveys,https://www.qmee.com/en-gb/surveys"
    ) == [
        "https://www.qmee.com/en-gb/surveys",
        "https://fallback.test/surveys",
    ]


def test_provider_rotates_only_after_committed_step_limit(monkeypatch):
    monkeypatch.setenv("SURVEY_PROVIDER_ENTRY_STEP_LIMIT", "25")
    assert not should_rotate_survey_provider({
        "step_number": 24,
        "survey_provider_start_step": 0,
        "survey_provider_question_started": False,
    })
    assert should_rotate_survey_provider({
        "step_number": 25,
        "survey_provider_start_step": 0,
        "survey_provider_question_started": False,
    })
    assert not should_rotate_survey_provider({
        "step_number": 100,
        "survey_provider_start_step": 0,
        "survey_provider_question_started": True,
    })


def test_timeout_heavy_provider_rotates_on_wall_clock_before_25_steps(monkeypatch):
    monkeypatch.setenv("SURVEY_PROVIDER_ENTRY_STEP_LIMIT", "25")
    monkeypatch.setenv("SURVEY_PROVIDER_ENTRY_TIMEOUT_SECONDS", "300")
    assert should_rotate_survey_provider({
        "step_number": 4,
        "survey_provider_start_step": 0,
        "survey_provider_started_at": time.time() - 301,
        "survey_provider_question_started": False,
    })
    assert not should_rotate_survey_provider({
        "step_number": 4,
        "survey_provider_start_step": 0,
        "survey_provider_started_at": time.time() - 301,
        "survey_provider_question_started": True,
    })


def test_stuck_timeout_requires_unchanged_semantic_page_state(monkeypatch):
    monkeypatch.setenv("SURVEY_STUCK_TIMEOUT_SECONDS", "180")
    base = {
        "survey_stuck_page_identity": survey_semantic_page_identity(
            "https://survey.test/q?token=old", "fingerprint-one"
        ),
        "survey_stuck_since": 1000.0,
        "survey_verified_progress_step": 11,
        "survey_stuck_progress_step": 11,
    }
    timed_out = survey_stuck_watch_updates(
        base,
        current_url="https://survey.test/q?token=new",
        page_fingerprint="fingerprint-one",
        interaction_fingerprint="form-one",
        active=True,
        now=1180.1,
    )
    assert timed_out["survey_stuck_timed_out"] is True

    changed_page = survey_stuck_watch_updates(
        base,
        current_url="https://survey.test/q",
        page_fingerprint="fingerprint-two",
        active=True,
        now=1180.1,
    )
    assert changed_page["survey_stuck_timed_out"] is False
    assert changed_page["survey_stuck_since"] == 1180.1

    # A commit marker alone is not enough: an inert/mislabelled click can be
    # mechanically successful. Only a changed route/question resets time.
    false_progress = survey_stuck_watch_updates(
        {**base, "survey_verified_progress_step": 12},
        current_url="https://survey.test/q?token=newer",
        page_fingerprint="fingerprint-one",
        interaction_fingerprint="form-one",
        active=True,
        now=1180.1,
    )
    assert false_progress["survey_stuck_timed_out"] is True

    # Changing only answer/control state is not a new question. Otherwise a
    # checkbox toggle loop can reset this deadline forever.
    form_churn = survey_stuck_watch_updates(
        base,
        current_url="https://survey.test/q?token=newer",
        page_fingerprint="fingerprint-one",
        interaction_fingerprint="form-two",
        active=True,
        now=1180.1,
    )
    assert form_churn["survey_stuck_timed_out"] is True
    assert form_churn["survey_stuck_since"] == 1000.0


def test_continuous_router_never_finalizes_after_recovery_cap():
    assert route_to_worker({
        "continuous_survey_mode": True,
        "step_number": 200,
        "max_steps": 25,
        "error_count": 8,
        "recovery_count": 20,
        "plan_steps": [],
    }) == "recovery"


def test_verified_answer_commit_starts_provider_question_and_resets_clock(monkeypatch):
    monkeypatch.setattr(brain_graph, "_PRM_CRITIC", None)
    state = BrainState(
        objective="Complete surveys",
        continuous_survey_mode=True,
        step_number=7,
        survey_home_url="https://www.qmee.com/en-gb/surveys",
        current_url="https://external-provider.test/questionnaire",
        proposed_action={
            "verb": "click",
            "target_name": "Next",
            "question_text": "Which option applies to you?",
            "answer_basis": "profile_fact",
        },
        action_outcome="→ OK (DOM changed)",
        history=[{"survey_transition_verified": True}],
    )

    updates = asyncio.run(brain_graph.commit_node(state))

    assert updates["survey_provider_question_started"] is True
    assert updates["survey_verified_progress_step"] == 8
    assert updates["survey_stuck_page_identity"] == ""
    assert updates["survey_stuck_timed_out"] is False
    assert updates["survey_stuck_elapsed_seconds"] == 0.0
    assert updates["survey_stuck_since"] > 0


def test_unverified_dispatch_never_starts_provider_question(monkeypatch):
    monkeypatch.setattr(brain_graph, "_PRM_CRITIC", None)
    state = BrainState(
        objective="Complete surveys",
        continuous_survey_mode=True,
        step_number=7,
        survey_home_url="https://www.qmee.com/en-gb/surveys",
        current_url="https://external-provider.test/questionnaire",
        proposed_action={
            "verb": "click",
            "target_name": "Next",
            "question_text": "Which option applies to you?",
            "answer_basis": "profile_fact",
        },
        action_outcome=(
            "→ DISPATCHED ONCE [verification pending]: click was sent but no "
            "observable effect was confirmed"
        ),
    )

    updates = asyncio.run(brain_graph.commit_node(state))

    assert "survey_verified_progress_step" not in updates
    assert "survey_provider_question_started" not in updates


def test_abandon_commit_rotates_provider_and_clears_cycle_context(monkeypatch):
    monkeypatch.setattr(brain_graph, "_PRM_CRITIC", None)
    providers = [
        "https://www.qmee.com/en-gb/surveys",
        "https://www.surveystreak.com/?page=dashboard",
    ]
    state = BrainState(
        objective="Complete surveys",
        continuous_survey_mode=True,
        step_number=25,
        survey_provider_urls=providers,
        survey_provider_index=0,
        survey_boundary_reason="provider_rotated:entry_step_limit",
        survey_boundary_target_url=providers[1],
        history=[{"step": 24, "action": "click", "outcome": "old context"}],
        survey_cycle_answers=[{"question_text": "Old", "answer_value": "Yes"}],
        proposed_action={
            "verb": "abandon_survey",
            "url": providers[1],
            "target_name": "survey provider dashboard",
            "survey_boundary_reason": "provider_rotated:entry_step_limit",
        },
        action_outcome="→ OK (survey closed; provider restored)",
    )

    updates = asyncio.run(brain_graph.commit_node(state))

    assert updates["survey_provider_index"] == 1
    assert updates["survey_provider_start_step"] == 26
    assert updates["survey_context_resets"] == 1
    assert updates["survey_cycle_answers"] == []
    assert updates["history"][0]["action"] == "survey_cycle_boundary"
