import json
from datetime import UTC, datetime, timedelta

import pytest

from agent_first_browse.agent import graph as brain_graph
from agent_first_browse.browser import cdp_input
from agent_first_browse.actions import tools as mcp_tools
from agent_first_browse.survey import profile as survey_profile
from agent_first_browse.survey.context import prepare_survey_transaction
from agent_first_browse.workers.base import _survey_fast_path


def _profile(**demographics):
    return {
        "name": "default",
        "learning": {"mode": "synthetic_persona", "auto_expand": True},
        "demographics": demographics,
        "stable_facts": {},
        "personality": {},
        "learned_answers": {},
    }


def test_sanitizer_repairs_json_and_removes_profile_pollution(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        """{
          "active_profile": "default",
          "profiles": {"default": {
            "demographics": {
              "date_of_birth": "2000-01-02",
              "age": 25,
              "postcode": "ab1 2cd",
              "relationship_status": "single",
              "marital_status": "single"
            },
            "stable_facts": {"bad_control": "radio"},
            "learned_answers": {}
          }},
        }""",
        encoding="utf-8",
    )

    report = survey_profile.sanitize_profile_file(path, force=True)
    document = json.loads(path.read_text(encoding="utf-8"))
    active = document["profiles"]["default"]

    assert report["sanitized"] is True
    assert report["repaired_json"] is True
    assert active["demographics"]["postal_code"] == "AB1 2CD"
    assert "postcode" not in active["demographics"]
    assert "age" not in active["demographics"]
    assert "relationship_status" not in active["demographics"]
    assert "bad_control" not in active["stable_facts"]
    assert path.with_suffix(".json.last-good").exists()


def test_invalid_profile_blocks_factual_fallback(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"profiles": {broken', encoding="utf-8")

    active = survey_profile.load_active_profile(path)

    assert active.get("_profile_error")
    assert active.get("demographics") is None


def test_scheduled_sanitizer_runs_after_write_threshold(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    old = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    path.write_text(json.dumps({
        "profiles": {"default": {"demographics": {}, "stable_facts": {}, "learned_answers": {}}},
        "maintenance": {"last_sanitized_at": old, "profile_writes_since_sanitize": 2},
    }))
    monkeypatch.setattr(survey_profile, "PROFILE_SANITIZE_AFTER_WRITES", 2)
    monkeypatch.setattr(survey_profile, "PROFILE_SANITIZE_INTERVAL_HOURS", 99.0)

    report = survey_profile.sanitize_profile_file(path)

    assert report["sanitized"] is True
    assert json.loads(path.read_text())["maintenance"]["profile_writes_since_sanitize"] == 0


def test_postcode_and_birthdate_are_pinned_to_profile():
    profile = _profile(postal_code="AB1 2CD", date_of_birth="2000-01-02")
    postcode_map = {
        "e1": {"kind": "input", "name": "Postcode", "value": "ZZ9 9ZZ"},
    }
    guarded, note, violation = survey_profile.enforce_typed_profile_fact(
        {"verb": "type", "element_id": "e1", "text": "ZZ9 9ZZ", "question_text": "Your postcode"},
        profile,
        postcode_map,
        page_text="Your postcode",
    )
    assert not violation and note
    assert guarded["text"] == "AB1 2CD"
    assert guarded["force_retype"] is True

    dob_map = {"e2": {"kind": "input", "name": "Birthdate"}}
    dob, note, violation = survey_profile.enforce_typed_profile_fact(
        {"verb": "type", "element_id": "e2", "text": "1999-09-09", "question_text": "Birthdate"},
        profile,
        dob_map,
        page_text="Birthdate",
    )
    assert not violation and note
    assert dob["verb"] == "set_date_of_birth"
    assert dob["text"] == "2000-01-02"


def test_dob_fast_path_handles_segmented_or_alternative_calendar(monkeypatch):
    profile = _profile(date_of_birth="2000-01-02")
    monkeypatch.setattr(survey_profile, "load_active_profile", lambda: profile)
    selector_map = {
        "e1": {"kind": "input", "name": "Day", "control_type": "text"},
        "e2": {"kind": "select", "name": "Month", "control_type": "select"},
        "e3": {"kind": "input", "name": "Year", "control_type": "text"},
        "e4": {"kind": "button", "text": "Use alternative calendar", "control_type": "button"},
    }
    state = {
        "continuous_survey_mode": True,
        "selector_map": selector_map,
        "page_text": "Please enter your birthdate",
        "survey_profile": profile,
        "current_url": "https://survey.example/question",
    }

    action = _survey_fast_path(state, set())

    assert action["verb"] == "set_date_of_birth"
    assert action["text"] == "2000-01-02"
    assert action["answer_basis"] == "configured_profile_fact"


def test_profile_text_fast_path_yields_during_loop_recovery(monkeypatch):
    profile = _profile(occupation="Social Worker")
    monkeypatch.setattr(survey_profile, "load_active_profile", lambda: profile)
    state = {
        "continuous_survey_mode": True,
        "selector_map": {"e3": {
            "kind": "input", "tag": "INPUT", "control_type": "text",
            "name": "Occupation", "value": "",
        }},
        "page_text": "What is your occupation?",
        "survey_profile": profile,
        "current_url": "https://click.cpx-research.com/question",
        "stagnation_level": 1,
        "correction_context": "Loop detected: choose a different action",
    }
    assert _survey_fast_path(state, set()) is None


def test_factual_choice_is_redirected_to_configured_value():
    profile = _profile(marital_status="Never married")
    selector_map = {
        "e1": {"control_type": "radio", "choice_group": "status", "text": "Married"},
        "e2": {"control_type": "radio", "choice_group": "status", "text": "Single"},
    }

    guarded, note, violation = survey_profile.enforce_profile_choice(
        {"verb": "click", "element_id": "e1", "question_text": "What is your marital status?"},
        profile,
        selector_map,
    )

    assert not violation and note
    assert guarded["element_id"] == "e2"
    assert guarded["profile_update_key"] == "marital_status"


def test_exact_social_worker_maps_to_truthful_provider_category():
    profile = _profile(occupation="Social Worker")
    selector_map = {
        "e1": {"control_type": "option", "choice_group": "job", "text": "Accountancy professionals"},
        "e2": {"control_type": "option", "choice_group": "job", "text": "Social and welfare professionals"},
        "e3": {"control_type": "option", "choice_group": "job", "text": "Other"},
    }
    guarded, note, violation = survey_profile.enforce_profile_choice(
        {"verb": "click", "element_id": "e1", "question_text": "What is your occupation?"},
        profile, selector_map,
    )
    assert not violation and note
    assert guarded["element_id"] == "e2"
    assert guarded["profile_update_value"] == "Social Worker"


def test_configured_fact_confirmation_is_never_written(tmp_path):
    path = tmp_path / "profiles.json"
    document = {
        "schema_version": 5, "active_profile": "default",
        "profiles": {"default": _profile(occupation="Social Worker")},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    profile = survey_profile.load_active_profile(path)
    _, learned, note = survey_profile.commit_confirmed_survey_answer(profile, {
        "verb": "type", "text": "Social Worker", "question_text": "Occupation",
        "answer_basis": "configured_profile_fact", "profile_update_category": "demographic",
        "profile_update_key": "occupation", "profile_update_mode": "set",
        "profile_update_value": "Social Worker",
    }, path)
    assert learned is False and "already authoritative" in note
    assert json.loads(path.read_text(encoding="utf-8"))["profiles"]["default"].get("learned_answers", {}) == {}


def test_schema_v5_prunes_survey_junk_and_keeps_recoverable_archive(tmp_path):
    path = tmp_path / "profiles.json"
    document = {
        "schema_version": 4, "active_profile": "default",
        "profiles": {"default": {
            "learning": {"mode": "synthetic_persona", "auto_expand": True},
            "demographics": {"occupation": "Social Worker", "favourite_bank": "Example Bank"},
            "stable_facts": {"home_ownership": "Rent", "milk_brand": "Example Milk"},
            "personality": {"name": "test", "learned_preferences": {"phone": "Example"}},
            "learned_answers": {"abc": {"key": "favourite_bank", "value": "Example Bank"}},
        }},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    report = survey_profile.sanitize_profile_file(path, force=True)
    cleaned = json.loads(path.read_text(encoding="utf-8"))["profiles"]["default"]
    assert report["sanitized"] and report["removed"] >= 3
    assert cleaned["demographics"] == {"occupation": "Social Worker"}
    assert cleaned["stable_facts"] == {"home_ownership": "Rent"}
    assert cleaned["personality"] == {"name": "test"}
    assert cleaned["learned_answers"] == {}
    assert cleaned["learning"]["auto_expand"] is False
    assert path.with_suffix(path.suffix + ".pre-v5-backup").exists()


def test_profile_choice_guard_never_rewrites_forward_button_to_selected_radio():
    """Regression: DeepLight's NEXT inherited the country group label."""
    profile = _profile(country="United Kingdom")
    selector_map = {
        "e3": {
            "kind": "button", "control_type": "option",
            "choice_group": "country", "group_label": "Which country do you live in?",
            "text": "United Kingdom [selected]", "selected": True,
        },
        "e10": {
            "kind": "button", "control_type": "button", "text": "NEXT",
            "choice_group": "country",
            "group_label": "Which country do you live in?",
        },
    }
    action = {
        "verb": "click", "element_id": "e10",
        "question_text": "Which country do you live in?",
        "answer_basis": "page_navigation",
    }

    guarded, note, violation = survey_profile.enforce_profile_choice(
        action, profile, selector_map,
    )

    assert guarded == action
    assert note == ""
    assert violation == ""


def test_native_select_is_pinned_to_profile_option():
    profile = _profile(gender="Male")
    selector_map = {
        "e1": {
            "kind": "input",
            "tag": "SELECT",
            "name": "What Is Your Gender?",
            "value": "Select",
            "options": [
                {"value": "", "label": "Select"},
                {"value": "M", "label": "Male"},
                {"value": "F", "label": "Female"},
            ],
        },
    }

    guarded, note, violation = survey_profile.enforce_profile_choice(
        {"verb": "select_option", "element_id": "e1", "text": "Female",
         "question_text": "What Is Your Gender?"},
        profile,
        selector_map,
    )

    assert not violation and note
    assert guarded["element_id"] == "e1"
    assert guarded["text"] == "M"
    assert guarded["profile_update_value"] == "Male"


def test_native_select_without_option_metadata_still_fails_closed_to_profile():
    profile = _profile(gender="Male")
    selector_map = {
        "e1": {"kind": "input", "tag": "SELECT", "name": "Gender", "value": "Select"},
    }

    guarded, note, violation = survey_profile.enforce_profile_choice(
        {"verb": "select_option", "element_id": "e1", "text": "Female",
         "question_text": "What Is Your Gender?"},
        profile,
        selector_map,
    )

    assert not violation and note
    assert guarded["text"] == "Male"


def test_native_profile_select_uses_deterministic_fast_path(monkeypatch):
    profile = _profile(gender="Male", date_of_birth="2000-01-02")
    monkeypatch.setattr(survey_profile, "load_active_profile", lambda: profile)
    state = {
        "continuous_survey_mode": True,
        "selector_map": {
            "e1": {
                "kind": "input", "tag": "SELECT", "name": "Gender",
                "value": "Select", "options": ["Select", "Male", "Female"],
            },
        },
        "page_text": "What Is Your Gender?",
        "survey_profile": profile,
        "current_url": "https://opinioninn.example/DemographyCheck.aspx",
    }

    action = _survey_fast_path(state, set())

    assert action["verb"] == "select_option"
    assert action["text"] == "Male"
    assert action["answer_basis"] == "configured_profile_fact"


def test_native_select_is_never_treated_as_a_text_input(monkeypatch):
    profile = _profile(age=20)
    monkeypatch.setattr(survey_profile, "load_active_profile", lambda: profile)
    state = {
        "continuous_survey_mode": True,
        "selector_map": {
            "e1": {
                "kind": "input", "tag": "SELECT", "name": "What is your age?",
                "value": "Select", "options": ["Select", "20", "21"],
            },
        },
        "page_text": "What is your age?",
        "survey_profile": profile,
        "current_url": "https://survey.test/prescreener",
    }

    action = _survey_fast_path(state, set())

    assert action["verb"] == "select_option"
    assert action["text"] == "20"


def test_native_select_reapplies_matching_value_after_validation_rejection(monkeypatch):
    profile = _profile(age=20)
    monkeypatch.setattr(survey_profile, "load_active_profile", lambda: profile)
    state = {
        "continuous_survey_mode": True,
        "selector_map": {
            "e1": {
                "kind": "input", "tag": "SELECT", "name": "What is your age?",
                "value": "20", "options": ["Select", "20", "21"],
            },
            "e2": {"kind": "button", "text": "Submit [disabled]", "disabled": True},
        },
        "page_text": "What is your age? Please select an item in the list.",
        "survey_profile": profile,
        "current_url": "https://survey.test/prescreener",
    }

    action = _survey_fast_path(state, set())

    assert action["verb"] == "select_option"
    assert action["element_id"] == "e1"
    assert action["text"] == "20"


def test_alias_profile_key_cannot_create_duplicate_fact():
    profile = _profile(education_level="GCSE")
    action = {
        "verb": "click",
        "answer_basis": "synthetic_profile_fact",
        "profile_update_category": "demographic",
        "profile_update_key": "education",
        "profile_update_mode": "set",
        "profile_update_value": "Degree",
        "profile_update_reason": "A proposed answer.",
        "target_name": "Degree",
        "question_text": "What is your highest education?",
    }

    violation = survey_profile.profile_learning_violation(action, profile)

    assert "already" in violation
    assert "education_level" in violation


def test_untrusted_queued_typing_is_rejected():
    selector_map = {
        "e1": {"kind": "button", "control_type": "radio", "text": "Male", "choice_group": "gender"},
        "e2": {"kind": "input", "control_type": "text", "name": "Postcode"},
    }
    queue, reason = prepare_survey_transaction(
        {"verb": "click", "element_id": "e1"},
        [{"verb": "type", "element_id": "e2", "text": "INVENTED"}],
        selector_map,
        page_text="Tell us about yourself",
        continuous_mode=True,
    )

    assert queue == []
    assert "authoritative configured profile fact" in reason


@pytest.mark.asyncio
async def test_typed_verification_is_bound_to_exact_element_id():
    class Page:
        def __init__(self):
            self.argument = None

        async def evaluate(self, _script, argument):
            self.argument = argument
            # e1 represents some other already-filled field; e2 is empty.
            return {"found": argument["expectedId"] == "e1", "text": "OTHER"}

    page = Page()
    result = await cdp_input._verify_typed_text(
        page, "EXPECTED", expected_element_id="e2"
    )

    assert page.argument == {"expectedId": "e2"}
    assert result["verified"] is False
    assert result["actual_length"] == 0


@pytest.mark.asyncio
async def test_date_widget_tool_executes_as_one_verified_operation(monkeypatch):
    class Page:
        def __init__(self):
            self.calls = []
            self.waits = []

        async def evaluate(self, script, argument=None):
            self.calls.append((script, argument))
            if argument is None:
                return False
            return {"ok": True, "mode": "segmented-date"}

        async def wait_for_timeout(self, milliseconds):
            self.waits.append(milliseconds)

    page = Page()
    monkeypatch.setattr(mcp_tools, "_get_page", lambda: page)

    result = await mcp_tools.mcp_set_date_of_birth("e2", "2000-01-02")

    assert result["success"] is True
    assert page.calls[-1][1] == {"id": "e2", "iso": "2000-01-02"}
    assert page.waits == [250]


@pytest.mark.asyncio
async def test_crashed_renderer_is_replaced_at_provider_dashboard(monkeypatch):
    class Replacement:
        def __init__(self, context):
            self.context = context
            self.url = "about:blank"

        def is_closed(self):
            return False

        def on(self, *_args):
            return None

        async def goto(self, url, **_kwargs):
            self.url = url

        async def bring_to_front(self):
            return None

    class Context:
        def __init__(self):
            self.pages = []

        async def new_page(self):
            page = Replacement(self)
            self.pages.append(page)
            return page

    class Crashed:
        def __init__(self, context):
            self.context = context
            self.url = "https://survey.example/question"

        def is_closed(self):
            return False

    context = Context()
    crashed = Crashed(context)
    context.pages.append(crashed)
    monkeypatch.setattr(mcp_tools, "_PAGE", crashed)
    mcp_tools._CRASHED_PAGE_IDS.add(id(crashed))

    result = await mcp_tools.recover_unusable_page(
        crashed, fallback_url="https://www.qmee.com/en-gb/surveys"
    )

    assert result["recovered"] is True
    assert result["page"] is not crashed
    assert result["page"].url == "https://www.qmee.com/en-gb/surveys"
    assert mcp_tools.get_page() is result["page"]


@pytest.mark.asyncio
async def test_perception_readiness_propagates_renderer_crash():
    class State:
        def model_dump(self):
            return {"objective": "Complete surveys"}

    class Page:
        url = "https://survey.example"

        async def evaluate(self, _script):
            raise RuntimeError("Page.evaluate: Target crashed")

    with pytest.raises(RuntimeError, match="Target crashed"):
        await brain_graph._wait_for_perception_readiness(Page(), State())
