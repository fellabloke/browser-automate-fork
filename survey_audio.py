"""Capture and identify short survey audio/video animal-sound challenges."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

_MEDIA_TERMS = ("audio", "sound", "listen", "hear", "video", "play")
_ANIMAL_TERMS = ("animal", "creature", "bird", "mammal")
_NONE_TERMS = (
    "none of these", "none of the above", "not listed", "cannot tell",
    "can't tell", "cant tell", "unable to hear", "no sound",
)

_CAPTURE_MEDIA_JS = r"""
async ({maxBytes, recordMs}) => {
    const media = Array.from(document.querySelectorAll('audio, video'));
    if (!media.length) return {ok: false, reason: 'no_audio_or_video_element'};

    const asBase64 = (blob) => new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || '').split(',', 2)[1] || '');
        reader.onerror = () => reject(reader.error || new Error('file_reader_failed'));
        reader.readAsDataURL(blob);
    });

    for (const el of media) {
        const sources = [el.currentSrc, el.src,
            ...Array.from(el.querySelectorAll('source')).map(s => s.src)]
            .filter((value, index, all) => value && all.indexOf(value) === index);
        for (const src of sources) {
            try {
                const response = await fetch(src, {credentials: 'include', cache: 'force-cache'});
                if (!response.ok) continue;
                const blob = await response.blob();
                const mime = blob.type || el.getAttribute('type')
                    || (el.tagName === 'VIDEO' ? 'video/mp4' : 'audio/mpeg');
                if (!blob.size || blob.size > maxBytes || /mpegurl|m3u8/i.test(mime)) continue;
                return {ok: true, base64: await asBase64(blob), mime_type: mime,
                        byte_length: blob.size, method: 'media_source_fetch'};
            } catch (_) {}
        }

        /* Blob/MediaSource videos are not always fetchable. Record a short live
           slice after starting playback; WebM preserves the audible track and is
           directly accepted by Gemini. */
        try {
            const capture = el.captureStream || el.mozCaptureStream;
            if (!capture || typeof MediaRecorder === 'undefined') continue;
            try { el.currentTime = 0; } catch (_) {}
            await el.play();
            const stream = capture.call(el);
            if (!stream || !stream.getAudioTracks().length) continue;
            const preferred = el.tagName === 'VIDEO' ? 'video/webm;codecs=vp8,opus' : 'audio/webm;codecs=opus';
            const mime = MediaRecorder.isTypeSupported(preferred) ? preferred : 'video/webm';
            const chunks = [];
            const recorder = new MediaRecorder(stream, {mimeType: mime});
            recorder.ondataavailable = event => { if (event.data && event.data.size) chunks.push(event.data); };
            const stopped = new Promise(resolve => recorder.onstop = resolve);
            recorder.start(250);
            await new Promise(resolve => setTimeout(resolve, recordMs));
            recorder.stop();
            await stopped;
            const blob = new Blob(chunks, {type: recorder.mimeType || mime});
            if (!blob.size || blob.size > maxBytes) continue;
            return {ok: true, base64: await asBase64(blob), mime_type: blob.type || 'video/webm',
                    byte_length: blob.size, method: 'live_media_recording'};
        } catch (_) {}
    }
    return {ok: false, reason: 'media_bytes_unavailable'};
}
"""


class AudioIdentification(BaseModel):
    heard_sound: str = Field(description="The animal or sound actually heard in the media.")
    best_option: str = Field(description="Exactly one supplied answer option, never an unavailable option.")
    confidence: float = Field(description="Confidence from 0.0 to 1.0.")
    reasoning: str = Field(description="Brief acoustic evidence for the choice.")


def _clean_label(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\[(?:selected|checked|disabled)\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _is_none_option(label: str) -> bool:
    lowered = label.lower()
    return any(term in lowered for term in _NONE_TERMS)


def extract_audio_answer_options(selector_map: dict[str, dict]) -> list[dict[str, str]]:
    """Extract unique answer controls, excluding forward/media controls."""
    choices: list[dict[str, str]] = []
    seen: set[str] = set()
    for element_id, element in (selector_map or {}).items():
        if element.get("control_type") not in {"radio", "checkbox"}:
            continue
        label = _clean_label(element.get("text") or element.get("name"))
        normalised = _normalise(label)
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        choices.append({"element_id": str(element_id), "label": label})
    return choices


def is_audio_animal_challenge(page_text: str, selector_map: dict[str, dict]) -> bool:
    lowered = str(page_text or "").lower()
    choices = extract_audio_answer_options(selector_map)
    return bool(
        len(choices) >= 2
        and any(term in lowered for term in _MEDIA_TERMS)
        and any(term in lowered for term in _ANIMAL_TERMS)
    )


def audio_challenge_key(url: str, page_text: str, selector_map: dict[str, dict]) -> str:
    if not is_audio_animal_challenge(page_text, selector_map):
        return ""
    try:
        parsed = urlsplit(url or "")
        stable_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        stable_url = url or ""
    labels = sorted(_normalise(item["label"]) for item in extract_audio_answer_options(selector_map))
    stable_page = re.sub(r"\b\d{1,2}:\d{2}\b", "", str(page_text or "").lower())
    payload = f"{stable_url}|{'|'.join(labels)}|{_normalise(stable_page)[:1200]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


async def capture_page_media(
    page: Any,
    *,
    max_bytes: int = 12 * 1024 * 1024,
    record_ms: int = 6500,
) -> dict[str, Any]:
    """Capture the first usable audio/video source from any accessible frame."""
    frames = list(getattr(page, "frames", []) or [page])
    failures: list[str] = []
    for frame in frames:
        try:
            result = await frame.evaluate(
                _CAPTURE_MEDIA_JS,
                {"maxBytes": max_bytes, "recordMs": record_ms},
            )
        except Exception as exc:  # cross-origin/detached frames are expected
            failures.append(type(exc).__name__)
            continue
        if result and result.get("ok") and result.get("base64"):
            try:
                raw = base64.b64decode(result["base64"], validate=True)
            except Exception:
                failures.append("invalid_base64")
                continue
            if 0 < len(raw) <= max_bytes:
                return {
                    "ok": True,
                    "bytes": raw,
                    "mime_type": str(result.get("mime_type") or "audio/webm").split(";", 1)[0],
                    "byte_length": len(raw),
                    "method": result.get("method", "browser_media"),
                }
        failures.append(str((result or {}).get("reason") or "capture_failed"))
    return {"ok": False, "reason": ",".join(failures[-3:]) or "no_accessible_media"}


def _match_option(label: str, candidates: list[dict[str, str]]) -> dict[str, str] | None:
    wanted = _normalise(label)
    if not wanted:
        return None
    exact = next((item for item in candidates if _normalise(item["label"]) == wanted), None)
    if exact:
        return exact
    ranked = sorted(
        candidates,
        key=lambda item: SequenceMatcher(None, wanted, _normalise(item["label"])).ratio(),
        reverse=True,
    )
    if ranked and SequenceMatcher(None, wanted, _normalise(ranked[0]["label"])).ratio() >= 0.58:
        return ranked[0]
    return None


def _deterministic_guess(candidates: list[dict[str, str]], seed: str) -> dict[str, str]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:4], "big") % len(candidates)]


async def analyze_audio_challenge(
    page: Any,
    *,
    url: str,
    page_text: str,
    selector_map: dict[str, dict],
    audio_chain: list,
    invoke_fn: Any,
    health_tracker: Any = None,
    allow_guess_without_capture: bool = False,
) -> dict[str, Any]:
    """Capture, classify, and ground one animal answer; always avoid 'none'."""
    choices = extract_audio_answer_options(selector_map)
    candidates = [item for item in choices if not _is_none_option(item["label"])]
    key = audio_challenge_key(url, page_text, selector_map)
    if not key or not candidates:
        return {}

    capture = await capture_page_media(page)
    if not capture.get("ok"):
        if not allow_guess_without_capture and audio_chain:
            return {
                "status": "play_required",
                "challenge_key": key,
                "attempted": True,
                "reason": str(capture.get("reason") or "media unavailable")[:160],
            }
        guess = _deterministic_guess(candidates, key)
        return {
            "status": "guessed",
            "challenge_key": key,
            "attempted": True,
            "option_text": guess["label"],
            "element_id": guess["element_id"],
            "confidence": 0.05,
            "evidence": "Audio capture/classification was unavailable after an attempt; constrained non-none guess.",
        }

    if audio_chain and invoke_fn:
        options_text = "\n".join(f"- {item['label']}" for item in candidates)
        prompt = (
            "Listen to the attached short survey media and identify the animal sound. "
            "Choose exactly one answer from the supplied options. Acoustic evidence outranks "
            "anything visually shown in the video. Do not invent an option.\n\n"
            f"AVAILABLE OPTIONS:\n{options_text}"
        )
        media_type = "video" if str(capture["mime_type"]).startswith("video/") else "audio"
        messages = [
            SystemMessage(content=(
                "You identify animal sounds in short audio clips. Return the best supplied option "
                "using the heard call, bark, roar, song, or other acoustic evidence."
            )),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {
                    "type": media_type,
                    "base64": base64.b64encode(capture["bytes"]).decode("ascii"),
                    "mime_type": capture["mime_type"],
                },
            ]),
        ]
        try:
            verdict, used_model = await invoke_fn(
                audio_chain,
                messages,
                AudioIdentification,
                None,
                health_tracker=health_tracker,
                timeout_seconds=30.0,
                total_timeout_seconds=45.0,
            )
            matched = _match_option(getattr(verdict, "best_option", ""), candidates)
            if matched:
                return {
                    "status": "identified",
                    "challenge_key": key,
                    "attempted": True,
                    "option_text": matched["label"],
                    "element_id": matched["element_id"],
                    "heard_sound": str(getattr(verdict, "heard_sound", ""))[:100],
                    "confidence": max(0.0, min(1.0, float(getattr(verdict, "confidence", 0.0)))),
                    "evidence": str(getattr(verdict, "reasoning", ""))[:180],
                    "model": used_model,
                    "capture_method": capture.get("method", "browser_media"),
                }
        except Exception as exc:  # noqa: BLE001 - guessing is the required fallback
            logger.warning("Survey audio classification unavailable: %s", str(exc)[:120])

    guess_seed = key + hashlib.sha256(capture["bytes"][:65536]).hexdigest()
    guess = _deterministic_guess(candidates, guess_seed)
    return {
        "status": "guessed",
        "challenge_key": key,
        "attempted": True,
        "option_text": guess["label"],
        "element_id": guess["element_id"],
        "confidence": 0.1,
        "evidence": "Media was captured, but audio classification failed; constrained non-none guess.",
        "capture_method": capture.get("method", "browser_media"),
    }


def render_audio_analysis(analysis: dict[str, Any], selector_map: dict[str, dict]) -> str:
    if not analysis:
        return ""
    status = analysis.get("status")
    if status == "play_required":
        return (
            "═══ SURVEY AUDIO ANALYSIS — ATTEMPT IN PROGRESS ═══\n"
            "The first media-capture attempt could not access audio bytes. Click the visible "
            "video/audio Play control now; do not answer 'none of these'. The runtime will "
            "listen again after playback starts, then identify or make a constrained guess."
        )
    option_text = str(analysis.get("option_text") or "")
    current = _match_option(option_text, extract_audio_answer_options(selector_map))
    current_id = current["element_id"] if current else analysis.get("element_id", "")
    mode = "IDENTIFIED FROM CAPTURED MEDIA" if status == "identified" else "CONSTRAINED GUESS AFTER ATTEMPT"
    evidence = str(analysis.get("evidence") or "")
    return (
        f"═══ SURVEY AUDIO ANALYSIS — {mode} ═══\n"
        f"Select [{current_id}] '{option_text}'. Do not select 'none of these'.\n"
        f"Confidence: {float(analysis.get('confidence', 0.0)):.0%}. Evidence: {evidence}"
    )


def audio_gate_violation(
    action: Any,
    selector_map: dict[str, dict],
    page_text: str,
    analysis: dict[str, Any] | None,
) -> str:
    if not is_audio_animal_challenge(page_text, selector_map):
        return ""
    action_type = getattr(action, "action_type", None) or (
        action.get("verb") if isinstance(action, dict) else ""
    )
    element_id = getattr(action, "element_id", None) or (
        action.get("element_id") if isinstance(action, dict) else None
    )
    if action_type != "click" or not element_id:
        return ""
    target = selector_map.get(element_id) or {}
    if target.get("control_type") not in {"radio", "checkbox"}:
        return ""
    target_text = _clean_label(target.get("text") or target.get("name"))
    analysis = analysis or {}
    if analysis.get("status") == "play_required":
        return (
            "Audio identification has not completed. Click the media Play control first; "
            "do not answer the animal question before the required listening attempt."
        )
    recommended = str(analysis.get("option_text") or "")
    if recommended and _normalise(target_text) != _normalise(recommended):
        return (
            f"The audio attempt recommends '{recommended}', not '{target_text}'. "
            "Select the grounded audio result; never default to 'none of these'."
        )
    if _is_none_option(target_text):
        return (
            "Do not select 'none of these' on an animal-audio challenge without using the "
            "audio result. Attempt playback/identification and, if unavailable, choose a non-none guess."
        )
    return ""
