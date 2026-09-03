"""Vision-on-demand — the a11y-DOM ⇄ vision "thinker" toggle (V21 + V30 coord fallback).

PRINCIPLE (the user's instruction, made literal)
═════════════════════════════════════════════════
The agent operates on the ACCESSIBILITY DOM by default — it is fast (~16ms/
extract), cheap (text tokens), and sufficient for the vast majority of steps.
It "opens its eyes" (one screenshot → vision model) ONLY when it is genuinely
confused or needs visual confirmation, resolves THAT step, and immediately falls
back to the a11y DOM. Vision is never the default and never sticky.

This is a thinker, not a parrot: the worker SELF-ASSESSES whether the DOM is
enough (`needs_vision`), and the system adds objective safety-net triggers
(repeated ineffective clicks; the escalation ladder's vision rung) so it also
gets eyes when it's stuck but hasn't realised it.

The vision model is given the a11y element MAP alongside the screenshot, so its
answer comes back as a stable `element_id` (resolved via the V19 registry).

V30 COORDINATE FALLBACK: when the target element is genuinely absent from the
a11y tree (shadow DOM, custom web components), Vision may return pixel coordinates
instead. These coordinates are validated by a second vision check ("Look-Before-
You-Leap") before being sent to Overwatch, ensuring grounded, safe clicks.

If vision is unavailable or unsure, we keep the a11y decision (no regression).
"""

from __future__ import annotations

import logging
import os
import hashlib
import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from app.logger import get_logger
    logger = get_logger("vision_consult")
except ImportError:
    logger = logging.getLogger("vision_consult")


# Per-survey budget. It resets at each cycle boundary; a fixed budget of five
# was too small for long screeners containing several visual/custom widgets.
try:
    MAX_VISION_CONSULTS = max(1, int(os.getenv("MAX_VISION_CONSULTS", "12")))
except (TypeError, ValueError):
    MAX_VISION_CONSULTS = 12
# Ineffective-action streak on the SAME target that forces a visual look even if
# the worker didn't ask (the DOM says "click" but the page never responds).
# One action with no observable progress is enough to get a fresh visual read.
# Waiting for a second identical attempt was the source of long blind loops.
INEFFECTIVE_STREAK_TRIGGER = 1
# Below this vision confidence we keep the a11y decision rather than override it.
MIN_VISION_CONFIDENCE = 0.55


def _positive_seconds(env_name: str, default: float) -> float:
    try:
        return max(1.0, float(os.getenv(env_name, str(default))))
    except (TypeError, ValueError):
        return default


# Vision is an optional disambiguation aid, so it must never stall the critical
# path for a minute while every free key times out. These remain configurable
# for unusually slow self-hosted or premium endpoints.
VISION_MODEL_TIMEOUT_SECONDS = _positive_seconds("VISION_MODEL_TIMEOUT_SECONDS", 20.0)
VISION_FAILOVER_BUDGET_SECONDS = _positive_seconds("VISION_FAILOVER_BUDGET_SECONDS", 45.0)
VISION_TIMEOUT_COOLDOWN_SECONDS = _positive_seconds("VISION_TIMEOUT_COOLDOWN_SECONDS", 45.0)
VISION_CACHE_TTL_SECONDS = _positive_seconds("VISION_CACHE_TTL_SECONDS", 30.0)
_VISION_CACHE: dict[str, tuple[float, dict[str, Any], str]] = {}


class VisionVerdict(BaseModel):
    """The vision model's grounded read of the screenshot, mapped back to an
    action on the labelled a11y elements (stable id, not pixels).

    V30: When the target is genuinely absent from the element map (shadow DOM,
    custom web component), coord_x/coord_y provide pixel-coordinate fallback."""

    situation: str = Field(
        default="normal_progress",
        description=(
            "Exactly one of: normal_progress, blocked_by_missing_target, "
            "blocked_by_validation, blocked_by_overlay, wrong_page, or "
            "goal_already_complete."
        )
    )
    scene_summary: str = Field(
        default="",
        description=(
            "Compact but detailed inventory of the viewport: page/popup, question "
            "or instruction, selected/input state, overlays, and relevant controls, "
            "including each control's approximate screen location."
        )
    )
    observation: str = Field(
        description="The strongest visual evidence relevant to the ambiguity."
    )
    blockage: str = Field(
        default="none",
        description=(
            "Explain WHY progress stopped, or write 'none'. Consider premature Next, "
            "unmet validation, disabled control, overlay, wrong target, missing DOM, "
            "or a page transition still loading."
        )
    )
    completed_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of what the agent visibly completed (maximum 4)."
    )
    next_step: str = Field(
        default="Follow the grounded action identified from the screenshot.",
        description=(
            "The single best next browser action, including the target's visible "
            "label and location. Do not suggest multiple speculative actions."
        )
    )
    target_description: str = Field(
        default="",
        description="Exact visible label, appearance, and location of the next target."
    )
    action_type: Literal[
        "click", "type", "scroll", "press_enter", "drag_and_drop",
        "wait", "done", "none"
    ] = Field(
        description=(
            "The action to take now, refined by what you SEE: one of "
            "'click','type','scroll','press_enter','drag_and_drop','wait','done','none'. "
            "Use 'none' if the screenshot does not change the a11y plan."
        )
    )
    element_id: str | None = Field(
        default=None,
        description="The element id (e.g. 'e7') from the ELEMENT MAP to act on.",
    )
    coord_x: float | None = Field(
        default=None,
        description=(
            "COORDINATE FALLBACK — only when the target is clearly visible on the "
            "screenshot but genuinely ABSENT from the ELEMENT MAP (e.g. shadow DOM "
            "button, custom web component). Return the center X pixel coordinate "
            "of the visual target. Leave null when element_id is provided."
        ),
    )
    coord_y: float | None = Field(
        default=None,
        description=(
            "COORDINATE FALLBACK — only when the target is clearly visible on the "
            "screenshot but genuinely ABSENT from the ELEMENT MAP. Return the center "
            "Y pixel coordinate of the visual target. Leave null when element_id "
            "is provided."
        ),
    )
    target_element_id: str | None = Field(
        default=None,
        description=(
            "For drag_and_drop, destination element-map id; leave null when it is "
            "only visible in the screenshot."
        ),
    )
    target_x: float | None = Field(
        default=None,
        description="For drag_and_drop, destination center X in screenshot pixels.",
    )
    target_y: float | None = Field(
        default=None,
        description="For drag_and_drop, destination center Y in screenshot pixels.",
    )
    text: str | None = Field(default=None, description="Text to type, if action_type='type'.")
    reasoning: str = Field(description="Why the screenshot leads to this action.")
    confidence: float = Field(description="0.0-1.0 — how sure you are from the image.")


VISION_SYSTEM_PROMPT = """You are the VISUAL CORTEX of a browser agent. The agent normally works from
the accessibility DOM (text), but it just hit something it cannot resolve from
text alone and switched to you for ONE look at the screen.

You are given: the objective, the question/ambiguity that triggered this look,
the agent's accessibility ELEMENT MAP (labelled e1,e2,… with positions), and a
SCREENSHOT of the current viewport.

Your job is an observer diagnosis, not a guess:
1. Inspect the ENTIRE screenshot from top to bottom. Inventory the current page,
   popup, question/instruction, selected values, validation messages, overlays,
   and all relevant controls. A large blue Next arrow at the bottom is a valid
   target even when its accessible label is vague or missing.
2. Explain WHY the agent is blocked. Decide whether it clicked Next too early,
   left an input unmet, hit validation, lost the target, encountered an overlay,
   or is simply on a different page.
3. Give a short ordered list of what has visibly happened and exactly ONE next
   step. Do not invent a control that is not visible.
4. Map what you see back to the ELEMENT MAP — return the element_id of the thing
   to act on. Correlate by label text and position.
5. DRAG ACTIONS: For a visible drag-and-drop task, return action_type='drag_and_drop'.
   Identify BOTH endpoints: put the draggable source in element_id or coord_x/coord_y,
   and put the drop destination in target_element_id or target_x/target_y. Never use
   'click' to begin a drag; the executor performs mouse-down, movement, and mouse-up.
6. COORDINATE FALLBACK: If the visual target is clearly visible on the screenshot
   but does NOT appear in the ELEMENT MAP at all (e.g. a shadow DOM button, custom
   web component, or element hidden from the accessibility tree), set element_id
   to null and instead return the target's CENTER pixel coordinates as coord_x and
   coord_y. The coordinate frame matches the screenshot: (0,0) is at the TOP-LEFT,
   x increases rightward, y increases downward. Use this ONLY when you are certain
   the element is genuinely absent from the map — not just hard to match.
7. Catch what text-only perception misses: which of several look-alike controls
   is the REAL/primary one, whether a prior click actually took effect, whether
   an overlay/popup is covering the target, whether the goal is visually already
   done (a confirmation/toast/cart badge), or whether the layout means the target
   is elsewhere.
8. If the screenshot does NOT change the plan, return action_type='none' — do not
   invent work. Be decisive and ground every claim in what is visibly on screen.
"""


def should_consult_vision(
    *,
    needs_vision: bool,
    state: dict,
    action_type: str = "",
) -> tuple[bool, str]:
    """Decide whether THIS step earns a vision look. Returns (consult, reason).

    Triggers (any one), all gated by the per-task budget:
      1. the worker self-flagged `needs_vision` (the thinker's own call);
      2. the escalation ladder reached its 'vision' rung (state.force_vision);
      3. an ineffective-action streak on the same target (the DOM says act, the
         page doesn't respond) — a classic "the element isn't what the DOM thinks"
         case that only a screenshot disambiguates.
    """
    used = int(state.get("vision_consults", 0) or 0)
    if used >= MAX_VISION_CONSULTS:
        return False, f"budget exhausted ({used}/{MAX_VISION_CONSULTS})"

    if needs_vision:
        return True, "worker requested visual confirmation"
    if state.get("force_vision"):
        return True, "escalation ladder vision rung"
    streak = int(state.get("ineffective_streak", 0) or 0)
    if streak >= INEFFECTIVE_STREAK_TRIGGER:
        return True, f"{streak} ineffective actions on the same target"
    # A routine loading wait needs no screenshot. Explicit uncertainty and
    # forced recovery above must still be allowed to open the agent's eyes.
    if action_type == "wait":
        return False, "routine wait action — no visual gain"
    return False, ""


def _build_vision_messages(objective, question, a11y_markdown, history_tail, base64_image,
                           vp_w=0, vp_h=0):
    """Build the multimodal message list (image attached via the model layer's
    base64_image path, mirroring advanced_agent's vision call)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    bounds = (
        f"═══ VIEWPORT BOUNDS (strict) ═══\n"
        f"The screenshot IS the browser viewport and NOTHING else — {vp_w}×{vp_h} px. "
        f"The only valid coordinate frame is x∈[0,{vp_w}], y∈[0,{vp_h}] with (0,0) at "
        f"the TOP-LEFT of the image. There is no browser chrome, taskbar, or desktop "
        f"in this image — ignore any such notion. Never reference anything outside "
        f"these bounds.\n\n"
        if vp_w and vp_h else ""
    )
    user = (
        f"═══ OBJECTIVE ═══\n{objective}\n\n"
        f"═══ WHY YOU'RE LOOKING (the ambiguity) ═══\n{question or 'Resolve the current step visually.'}\n\n"
        + bounds
        + f"═══ RECENT ACTIONS (short context) ═══\n{str(history_tail or '(none)')[-1600:]}\n\n"
        f"═══ ACCESSIBILITY ELEMENT MAP (map what you see to these ids) ═══\n"
        f"{a11y_markdown[:4000]}\n\n"
        "Look at the attached screenshot and return the complete observer diagnosis: "
        "situation, scene_summary, observation, blockage, completed_steps, next_step, "
        "target_description, then the grounded action."
    )
    return [SystemMessage(content=VISION_SYSTEM_PROMPT), HumanMessage(content=user)]


async def consult_vision(
    invoke_fn,
    vision_chain: list,
    breaker,
    health_tracker,
    *,
    objective: str,
    question: str,
    a11y_markdown: str,
    history_tail: str = "",
    allow_cache: bool = True,
) -> tuple[VisionVerdict | None, str]:
    """Capture one screenshot and ask the vision model to resolve the step.

    Returns (verdict, used_model). verdict is None when vision is unavailable,
    the screenshot fails, or the model can't answer — caller keeps the a11y
    decision (no regression). Never raises.
    """
    if invoke_fn is None or not vision_chain:
        return None, ""

    consult_started = time.monotonic()
    logger.info(
        "👁️ VISION REQUEST: reason=%s | question=%s | chain=%d | failover_budget=%.1fs | per_model_cap=%.1fs",
        str(question or "not specified")[:240],
        str(question or "not specified")[:500],
        len(vision_chain),
        VISION_FAILOVER_BUDGET_SECONDS,
        VISION_MODEL_TIMEOUT_SECONDS,
    )
    from mcp_tools import mcp_screenshot
    shot = await mcp_screenshot(full_page=False)
    if not shot.get("ok"):
        logger.warning(
            "👁️ VISION REQUEST FAILED before model call: screenshot=%s error=%s",
            False, str(shot.get("error") or "unknown")[:240],
        )
        return None, ""

    logger.info(
        "👁️ VISION SCREENSHOT READY: viewport=%sx%s image_bytes≈%d map_chars=%d history_chars=%d",
        shot.get("width", 0), shot.get("height", 0),
        len(str(shot.get("base64") or "")),
        len(str(a11y_markdown or "")), len(str(history_tail or "")),
    )

    cache_key = hashlib.sha256(
        "\n".join((
            str(question or "")[:1000],
            str(a11y_markdown or "")[:5000],
            str(shot.get("base64") or ""),
        )).encode("utf-8")
    ).hexdigest()
    if allow_cache:
        cached = _VISION_CACHE.get(cache_key)
        if cached and time.monotonic() - cached[0] <= VISION_CACHE_TTL_SECONDS:
            cached_verdict = VisionVerdict.model_validate(cached[1])
            logger.info("👁️ Vision cache hit for unchanged screenshot")
            logger.info(
                "👁️ VISION RESPONSE (cache:%s, %.1fs): %s",
                cached[2], time.monotonic() - consult_started,
                json.dumps(cached_verdict.model_dump(), ensure_ascii=False, default=str)[:6000],
            )
            return cached_verdict, cached[2]

    messages = _build_vision_messages(
        objective, question, a11y_markdown, history_tail, shot["base64"],
        vp_w=shot.get("width", 0), vp_h=shot.get("height", 0),
    )
    try:
        verdict, used_model = await invoke_fn(
            vision_chain, messages, VisionVerdict,
            breaker, base64_image=shot["base64"], health_tracker=health_tracker,
            timeout_seconds=VISION_MODEL_TIMEOUT_SECONDS,
            total_timeout_seconds=VISION_FAILOVER_BUDGET_SECONDS,
            timeout_cooldown_seconds=VISION_TIMEOUT_COOLDOWN_SECONDS,
        )
        if verdict is None:
            return None, used_model
        if allow_cache and float(verdict.confidence or 0.0) >= MIN_VISION_CONFIDENCE:
            if len(_VISION_CACHE) >= 32:
                oldest = min(_VISION_CACHE, key=lambda key: _VISION_CACHE[key][0])
                _VISION_CACHE.pop(oldest, None)
            _VISION_CACHE[cache_key] = (
                time.monotonic(), verdict.model_dump(), used_model,
            )
        coord_info = ""
        if not verdict.element_id and verdict.coord_x is not None and verdict.coord_y is not None:
            coord_info = f" at ({verdict.coord_x:.0f},{verdict.coord_y:.0f})"
        logger.info(
            "👁️ Vision consult (%s): situation=%s blockage=%s next=%s → %s%s%s [%.0f%%]",
            used_model, verdict.situation[:30], verdict.blockage[:80],
            verdict.next_step[:90], verdict.action_type,
            f" {verdict.element_id}" if verdict.element_id else "",
            coord_info,
            float(verdict.confidence) * 100,
        )
        # Keep the complete structured diagnosis available for post-run
        # analysis. The screenshot/base64 payload is deliberately excluded.
        response_json = json.dumps(
            verdict.model_dump(), ensure_ascii=False, default=str,
        )
        logger.info(
            "👁️ VISION RESPONSE (%s, %.1fs): %s",
            used_model, time.monotonic() - consult_started, response_json[:6000],
        )
        return verdict, used_model
    except Exception as e:
        logger.warning(
            "👁️ VISION REQUEST END: unavailable after %.1fs (%s) — keeping a11y decision",
            time.monotonic() - consult_started, str(e)[:240],
        )
        return None, ""


def apply_vision_verdict(proposed: dict, verdict: VisionVerdict) -> tuple[dict, bool]:
    """Merge a confident vision verdict into the a11y proposed action.

    Returns (proposed, overridden). We override only when the vision model is
    confident AND proposes a concrete action — otherwise the a11y decision stands
    (vision confirmed it, or had nothing to add).

    V30: Handles coordinate fallback — when Vision returns coord_x/coord_y
    instead of element_id (target absent from a11y tree), the coordinates are
    set on the proposed action with a 'vision_coords' flag for downstream
    validation before Overwatch grounding.
    """
    if verdict is None:
        return proposed, False
    if float(verdict.confidence) < MIN_VISION_CONFIDENCE:
        return proposed, False
    if verdict.action_type in ("", "none"):
        return proposed, False

    original = dict(proposed)
    proposed = dict(proposed)
    proposed["verb"] = verdict.action_type
    is_drag = verdict.action_type == "drag_and_drop"
    if verdict.element_id:
        proposed["element_id"] = verdict.element_id
        # Coordinates are re-resolved from the id by the V19 registry at action
        # time; drop stale a11y coords so they can't fight the vision choice.
        proposed["x"] = None
        proposed["y"] = None
        proposed["vision_coords"] = False
    elif (getattr(verdict, "coord_x", None) is not None
          and getattr(verdict, "coord_y", None) is not None):
        # V30 COORDINATE FALLBACK: target not in a11y tree — Vision returned
        # pixel coordinates. Clear element_id and set coords for downstream
        # validation ("Look-Before-You-Leap" in the worker pipeline).
        proposed["element_id"] = None
        proposed["x"] = float(verdict.coord_x)
        proposed["y"] = float(verdict.coord_y)
        proposed["vision_coords"] = True
        logger.info(
            "👁️📍 Vision coordinate fallback: target at (%.0f, %.0f) — "
            "element not in a11y tree",
            verdict.coord_x, verdict.coord_y,
        )
    if is_drag:
        # Drag actions have two independently grounded endpoints. Preserve the
        # destination fields instead of letting the single-target merge reduce
        # the visual instruction to a click.
        proposed["target_element_id"] = verdict.target_element_id
        proposed["target_x"] = verdict.target_x
        proposed["target_y"] = verdict.target_y
        if verdict.target_element_id:
            proposed["target_x"] = None
            proposed["target_y"] = None
        elif verdict.target_x is None or verdict.target_y is None:
            logger.warning("Vision proposed drag without a grounded destination")
            return original, False
        if not verdict.element_id and (verdict.coord_x is None or verdict.coord_y is None):
            logger.warning("Vision proposed drag without a grounded source")
            return original, False
        # Coordinate validation below must cover either endpoint when vision
        # supplied raw pixels rather than an element id.
        if verdict.target_x is not None and verdict.target_y is not None:
            proposed["vision_coords"] = True
    if verdict.text is not None:
        proposed["text"] = verdict.text
    proposed["reasoning"] = (
        f"[vision] {verdict.reasoning[:120]} "
        f"Diagnosis: {verdict.blockage[:120]}. Next: {verdict.next_step[:120]}"
    )
    proposed["vision_used"] = True
    return proposed, True
