"""Regression tests for survey audio capture, identification, and fallback gates."""

from __future__ import annotations

import asyncio
import base64

from agent_first_browse.agent.state import BrainState
from agent_first_browse.survey.audio import (
    _CAPTURE_MEDIA_JS,
    AudioIdentification,
    analyze_audio_challenge,
    audio_challenge_key,
    capture_page_media,
    extract_audio_answer_options,
    is_audio_animal_challenge,
)
from agent_first_browse.survey.context import build_survey_handoff, survey_gate_violation

PAGE_TEXT = "Click Play, listen to the sound, then select which animal you hear."
SELECTOR_MAP = {
    "e1": {"kind": "button", "text": "Play video"},
    "e2": {"kind": "button", "text": "Dog", "control_type": "radio"},
    "e3": {"kind": "button", "text": "Lion", "control_type": "radio"},
    "e4": {"kind": "button", "text": "None of these", "control_type": "radio"},
    "e5": {"kind": "button", "text": "Next"},
}


def test_detects_animal_audio_question_and_excludes_none_from_candidates():
    assert is_audio_animal_challenge(PAGE_TEXT, SELECTOR_MAP)
    assert audio_challenge_key("https://survey.example/q1", PAGE_TEXT, SELECTOR_MAP)
    assert [item["label"] for item in extract_audio_answer_options(SELECTOR_MAP)] == [
        "Dog", "Lion", "None of these",
    ]
    assert "captureStream" in _CAPTURE_MEDIA_JS
    assert "MediaRecorder" in _CAPTURE_MEDIA_JS


def test_browser_media_capture_decodes_bounded_base64():
    class Frame:
        async def evaluate(self, _script, _args):
            return {
                "ok": True,
                "base64": base64.b64encode(b"fake-mp3").decode("ascii"),
                "mime_type": "audio/mp3",
                "method": "media_source_fetch",
            }

    class Page:
        frames = [Frame()]

    result = asyncio.run(capture_page_media(Page()))

    assert result["ok"] is True
    assert result["bytes"] == b"fake-mp3"
    assert result["mime_type"] == "audio/mp3"


def test_captured_audio_is_classified_and_grounded_to_visible_option(monkeypatch):
    async def fake_capture(_page):
        return {
            "ok": True,
            "bytes": b"animal-audio",
            "mime_type": "audio/mp3",
            "method": "media_source_fetch",
        }

    async def fake_invoke(chain, messages, schema, breaker, **kwargs):
        assert chain == ["audio-model"]
        assert schema is AudioIdentification
        assert messages[-1].content[1]["type"] == "audio"
        return AudioIdentification(
            heard_sound="A dog barking",
            best_option="Dog",
            confidence=0.94,
            reasoning="Repeated short barks",
        ), "gemini-audio:test"

    monkeypatch.setattr("agent_first_browse.survey.audio.capture_page_media", fake_capture)
    result = asyncio.run(analyze_audio_challenge(
        object(),
        url="https://survey.example/q1",
        page_text=PAGE_TEXT,
        selector_map=SELECTOR_MAP,
        audio_chain=["audio-model"],
        invoke_fn=fake_invoke,
    ))

    assert result["status"] == "identified"
    assert result["option_text"] == "Dog"
    assert result["element_id"] == "e2"
    assert result["confidence"] == 0.94


def test_failed_first_capture_requests_play_then_second_attempt_guesses(monkeypatch):
    async def failed_capture(_page):
        return {"ok": False, "reason": "media_bytes_unavailable"}

    monkeypatch.setattr("agent_first_browse.survey.audio.capture_page_media", failed_capture)
    first = asyncio.run(analyze_audio_challenge(
        object(),
        url="https://survey.example/q1",
        page_text=PAGE_TEXT,
        selector_map=SELECTOR_MAP,
        audio_chain=["audio-model"],
        invoke_fn=object(),
        allow_guess_without_capture=False,
    ))
    second = asyncio.run(analyze_audio_challenge(
        object(),
        url="https://survey.example/q1",
        page_text=PAGE_TEXT,
        selector_map=SELECTOR_MAP,
        audio_chain=["audio-model"],
        invoke_fn=object(),
        allow_guess_without_capture=True,
    ))

    assert first["status"] == "play_required"
    assert second["status"] == "guessed"
    assert second["option_text"] in {"Dog", "Lion"}
    assert second["option_text"] != "None of these"


def test_audio_gate_blocks_none_and_enforces_identified_option():
    analysis = {
        "status": "identified",
        "option_text": "Lion",
        "element_id": "e3",
        "confidence": 0.9,
    }
    none_reason = survey_gate_violation(
        {"verb": "click", "element_id": "e4"},
        SELECTOR_MAP,
        page_text=PAGE_TEXT,
        audio_analysis=analysis,
    )
    wrong_reason = survey_gate_violation(
        {"verb": "click", "element_id": "e2"},
        SELECTOR_MAP,
        page_text=PAGE_TEXT,
        audio_analysis=analysis,
    )

    assert "Lion" in none_reason
    assert "Lion" in wrong_reason
    assert survey_gate_violation(
        {"verb": "click", "element_id": "e3"},
        SELECTOR_MAP,
        page_text=PAGE_TEXT,
        audio_analysis=analysis,
    ) == ""


def test_audio_handoff_orders_play_before_answer_and_exposes_result():
    play_handoff = build_survey_handoff({
        "objective": "Complete the survey",
        "page_text": PAGE_TEXT,
        "selector_map": SELECTOR_MAP,
        "survey_audio_analysis": {"status": "play_required", "attempted": True},
    })
    result_handoff = build_survey_handoff({
        "objective": "Complete the survey",
        "page_text": PAGE_TEXT,
        "selector_map": SELECTOR_MAP,
        "survey_audio_analysis": {
            "status": "identified",
            "option_text": "Dog",
            "element_id": "e2",
            "confidence": 0.9,
            "evidence": "Barking",
        },
    })

    assert "Click the visible video/audio Play control" in play_handoff
    assert "Select [e2] 'Dog'" in result_handoff
    assert "Do not select 'none of these'" in result_handoff
    assert "survey_audio_analysis" in BrainState.model_fields
