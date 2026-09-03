"""Long-session context and checkpoint retention regressions."""

from __future__ import annotations

import pytest
import sqlite3

import brain_graph
import mcp_tools
from brain_state import (
    HISTORY_MAX_ENTRIES,
    HISTORY_PROMPT_MAX_CHARS,
    SURVEY_CYCLE_ARCHIVE_MAX,
    BrainState,
    append_bounded,
)
from checkpoint_retention import prune_checkpoint_database
from survey_context import render_cycle_answer_memory, survey_cycle_cleanup_updates
from survey_profile import (
    PROFILE_PROMPT_MAX_CHARS,
    compact_runtime_profile,
    render_profile,
)


class _Page:
    url = "https://survey-dashboard.test/offers"

    async def wait_for_load_state(self, *_args, **_kwargs):
        return None

    async def wait_for_timeout(self, *_args, **_kwargs):
        return None


def test_accumulated_history_is_strictly_bounded():
    history: list[dict] = []
    for index in range(HISTORY_MAX_ENTRIES * 4):
        history = append_bounded(history, [{"step": index}], HISTORY_MAX_ENTRIES)

    assert len(history) == HISTORY_MAX_ENTRIES
    assert history[0]["step"] == HISTORY_MAX_ENTRIES * 3


def test_prompt_history_keeps_compact_survey_answers_not_all_raw_actions():
    history = [{
        "step": index,
        "verb": "click",
        "element_id": "e18",
        "target_name": f"Answer {index}",
        "answer_value": f"Answer {index}",
        "question_text": f"Question {index}: choose the consistent response",
        "outcome": "→ OK (structure changed)",
        "screen": "survey question",
    } for index in range(200)]
    rendered = BrainState(objective="Complete surveys", history=history).compress_history()

    assert len(rendered) <= HISTORY_PROMPT_MAX_CHARS + 50
    assert "CURRENT SURVEY ANSWER LEDGER" in rendered
    assert "Question 199" in rendered
    assert "Question 0" not in rendered
    assert rendered.count("Action turn") <= 8


def test_unbounded_profile_storage_becomes_bounded_question_relevant_runtime_state():
    learned = {
        f"answer-{index}": {
            "question": (
                "Which cat food brand did you buy?" if index == 0
                else f"Unrelated preference question {index}"
            ),
            "value": "Whiskers" if index == 0 else f"Value {index}",
        }
        for index in range(200)
    }
    preferences = {f"preference_{index}": f"value_{index}" for index in range(150)}
    preferences["cat_food_brand"] = "Whiskers"
    profile = {
        "name": "long-lived",
        "learning": {"mode": "synthetic_persona", "auto_expand": True},
        "demographics": {"age": 30},
        "personality": {"learned_preferences": preferences},
        "learned_answers": learned,
    }

    compact = compact_runtime_profile(profile, "What cat food brand have you purchased?")
    rendered = render_profile(profile, "What cat food brand have you purchased?")

    assert len(compact["learned_answers"]) < len(learned)
    assert "answer-0" in compact["learned_answers"]
    assert compact["personality"]["learned_preferences"]["cat_food_brand"] == "Whiskers"
    assert "Which cat food brand" in rendered
    assert len(rendered) <= PROFILE_PROMPT_MAX_CHARS


def test_long_survey_archive_retrieves_old_relevant_answer_with_bounded_output():
    archive = [{
        "question_text": (
            "Which cat food brand did the advert show?" if index == 0
            else f"Unrelated product rating {index}"
        ),
        "answer_value": "Whiskers" if index == 0 else str(index % 5),
    } for index in range(SURVEY_CYCLE_ARCHIVE_MAX)]

    rendered = render_cycle_answer_memory(
        archive,
        "Earlier, which cat food brand was shown in the advert?",
        max_chars=7000,
    )

    assert "Which cat food brand" in rendered
    assert "Whiskers" in rendered
    assert "Unrelated product rating 159" in rendered
    assert len(rendered) <= 7040


def test_cycle_cleanup_preserves_durable_identity_but_resets_local_reasoning():
    state = BrainState(
        objective="Complete surveys",
        continuous_survey_mode=True,
        survey_cycles_completed=3,
        survey_profile={"name": "fixed-persona", "demographics": {"age": 30}},
        history=[{"step": 40, "question_text": "Old question", "answer_value": "Old answer"}],
        survey_cycle_answers=[{"question_text": "Old question", "answer_value": "Old answer"}],
        loop_signatures=["old-loop"],
        beliefs=["old provider-specific lesson"],
        vision_consults=4,
        prm_checklist=[{"desc": "Complete survey", "status": "done", "verified": True}],
        plan_steps=[{"id": 1, "desc": "Open dashboard", "status": "done"}],
    )

    updates = survey_cycle_cleanup_updates(state.model_dump())
    resumed = state.model_copy(update=updates)

    assert resumed.survey_profile == state.survey_profile
    assert resumed.survey_cycles_completed == 3
    assert resumed.history[0]["action"] == "survey_cycle_boundary"
    assert "durable respondent profile preserved" in resumed.history[0]["outcome"]
    assert resumed.loop_signatures == [] and resumed.beliefs == []
    assert resumed.survey_cycle_answers == []
    assert resumed.vision_consults == 0
    assert resumed.prm_checklist[0]["status"] == "pending"
    assert resumed.plan_steps[0]["status"] == "active"


@pytest.mark.asyncio
async def test_perception_cleans_only_after_leaving_verified_completion(monkeypatch):
    page = _Page()

    async def active_page():
        return page

    async def snapshot():
        return {
            "elements": [],
            "markdown": "Survey offers dashboard",
            "page_text": "Available surveys",
            "selector_map": {},
            "element_count": 0,
        }

    async def no_login():
        return {"logged_in": False}

    monkeypatch.setattr(brain_graph, "_sync_active_page", active_page)
    monkeypatch.setattr(mcp_tools, "mcp_snapshot", snapshot)
    monkeypatch.setattr(mcp_tools, "mcp_detect_login", no_login)
    monkeypatch.setenv("V29_ADAPTIVE_PERCEPTION", "0")
    state = BrainState(
        objective="Complete surveys",
        continuous_survey_mode=True,
        survey_cycles_completed=1,
        survey_cycle_boundary_pending=True,
        last_survey_completion_signature="completion-page",
        survey_page_fingerprint="completion-page",
        history=[{"step": 30, "question_text": "Old", "answer_value": "Old"}],
        vision_consults=3,
    )

    updates = await brain_graph.perceive_node(state)

    assert updates["survey_cycle_boundary_pending"] is False
    assert updates["survey_context_resets"] == 1
    assert updates["history"][0]["action"] == "survey_cycle_boundary"
    assert "Old" not in updates["history_compressed"]
    assert updates["vision_consults"] == 0


def test_checkpoint_retention_keeps_active_window_and_recent_threads(tmp_path, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_RETENTION_ENABLED", raising=False)
    db_path = tmp_path / "checkpoints.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE checkpoints (
                thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT,
                type TEXT, checkpoint BLOB, metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
            CREATE TABLE writes (
                thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL,
                idx INTEGER NOT NULL, channel TEXT NOT NULL, type TEXT, value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
        """)
        rows = []
        for thread, prefix, count in (
            ("active", "300", 6),
            ("recent-prior", "200", 4),
            ("expired-prior", "100", 4),
        ):
            for index in range(count):
                rows.append((
                    thread, "", f"{prefix}-{index:03d}", None,
                    "json", b"{}", b"{}",
                ))
        conn.executemany(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()

        result = prune_checkpoint_database(
            db_path,
            "active",
            keep_active=3,
            keep_prior_threads=1,
            keep_per_prior_thread=1,
        )

        counts = dict(conn.execute(
            "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id"
        ).fetchall())

    assert counts == {"active": 3, "recent-prior": 1}
    assert result == {"checkpoints_deleted": 10, "threads_deleted": 1}
