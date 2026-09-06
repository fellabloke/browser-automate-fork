"""Startup and capability probing for already-constructed model clients.

This module owns startup probe orchestration and interpretation.  It does not
construct providers, choose ordinary inference routes, or perform failover.
The JSON-mode executor is imported from ``failover`` so normal inference and
probe rescue share one implementation without a probe-to-registry dependency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from .health import ProviderHealthTracker, base_model_name, normalize_model_id
from .schemas import ModelClient
from .failover import invoke_json_mode

try:
    from agent_first_browse.logging import get_logger

    logger = get_logger("model_registry")
except ImportError:
    logger = logging.getLogger("model_registry")


AGENTIC_TEXT_ALLOWLIST: set[str] = {
    "gemini-3.5-flash-lite",
    "gpt-oss-120b",
    "nemotron-3.5-lightning-30b-a3b",
    "llama-3.3-70b-instruct-fp8-fast",
    "gemma-4-31b-it",
    "gemma-4-32b-it",
    "gemma-4-26b-a4b-it",
    "llama-3.3-70b-instruct",
    "llama-3.3-nemotron-super-49b-v1.5",
}


def combo_of(model: ModelClient) -> tuple[str, str, str]:
    """Return the provider/model/pipeline identity used by probe gating."""
    return (model.provider, normalize_model_id(base_model_name(model.name)), model.pipeline)


def apply_capability_gate(
    pipeline: list[ModelClient],
    dead: set,
    incapable: set,
    label: str,
) -> list[ModelClient]:
    """Remove dead/incapable combinations while preserving the safety floor."""
    kept = [model for model in pipeline if combo_of(model) not in dead and combo_of(model) not in incapable]
    if kept:
        if len(kept) != len(pipeline):
            logger.info("🔬 Capability gate [%s]: %d → %d models", label, len(pipeline), len(kept))
        return kept
    alive = [model for model in pipeline if combo_of(model) not in dead]
    allow = [
        model
        for model in alive
        if normalize_model_id(base_model_name(model.name)) in AGENTIC_TEXT_ALLOWLIST
    ]
    floor = allow or alive or pipeline
    logger.warning("🛟 Capability gate would EMPTY %s — restoring %d floor model(s)", label, len(floor))
    return floor


async def probe_and_prune(
    text_pipeline: list[ModelClient],
    vision_pipeline: list[ModelClient],
    health: ProviderHealthTracker,
    *,
    timeout: float = 8.0,
    vision_timeout: float | None = None,
    probe_vision: bool = True,
) -> tuple[list[ModelClient], list[ModelClient]]:
    """Probe representative model combinations and return surviving pipelines."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel as _BM, Field as _F

    reps: dict[tuple[str, str, str], ModelClient] = {}
    probe_pipeline = text_pipeline + (vision_pipeline if probe_vision else [])
    for model in probe_pipeline:
        key = combo_of(model)
        current = reps.get(key)
        if current is None or (
            health.is_chronically_unreliable(current.name)
            and not health.is_chronically_unreliable(model.name)
        ):
            reps[key] = model

    cached_count = sum(1 for model in reps.values() if health.probe_cache_fresh(model.name))
    reps = {
        key: model
        for key, model in reps.items()
        if not health.probe_cache_fresh(model.name) and not health.is_hard_dead(model.name)
    }
    known_dead = {
        combo_of(model) for model in probe_pipeline if health.is_hard_dead(model.name)
    }
    if known_dead:
        text_pipeline = apply_capability_gate(text_pipeline, known_dead, set(), "TEXT")
        vision_pipeline = apply_capability_gate(vision_pipeline, known_dead, set(), "VISION")
        logger.info("🔬 Removed %d cached hard-dead model combination(s) before probing", len(known_dead))
    if cached_count:
        logger.info("🔬 Capability probe cache reused for %d healthy combo(s)", cached_count)
    if not reps:
        return text_pipeline, vision_pipeline

    class _CapProbe(_BM):
        action: str = _F(description="one of: click, type, scroll, done")
        element_id: str = _F(description="an element id like e5")
        confidence: float = _F(description="0..1")

    cap_messages = [
        SystemMessage(
            content="You are a browser agent. Choose the next action and "
            "reply with the structured fields."
        ),
        HumanMessage(
            content="Page elements: [e3] search box (empty), [e7] Submit "
            "button. Goal: type 'running shoes' into the search box. "
            "Give your next action."
        ),
    ]

    dead_markers = ("404", "401", "410")
    dead_text = (
        "not_found",
        "does not exist",
        "model_not_found",
        "invalid api key",
        "invalid_api_key",
        "unauthorized",
        "decommissioned",
        "gone",
    )

    def kind(exc: BaseException) -> str:
        value = str(exc)
        lowered = value.lower()
        if any(marker in value for marker in dead_markers) or any(marker in lowered for marker in dead_text):
            return "dead"
        if "429" in value or "rate limit" in lowered or "rate_limit" in lowered:
            return "transient"
        return "other"

    async def probe(combo_key, model: ModelClient):
        probe_timeout = (
            vision_timeout
            if model.pipeline == "vision" and vision_timeout is not None
            else timeout
        )
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                model.client.with_structured_output(_CapProbe).ainvoke(cap_messages),
                timeout=probe_timeout,
            )
            return combo_key, time.monotonic() - started, "capable", "strict"
        except asyncio.TimeoutError:
            return combo_key, time.monotonic() - started, "timeout", None
        except Exception as exc:
            status = kind(exc)
            if status == "dead":
                return combo_key, time.monotonic() - started, "dead", str(exc)[:120]
            if status == "transient":
                return combo_key, time.monotonic() - started, "transient", str(exc)[:100]
            try:
                await asyncio.wait_for(
                    invoke_json_mode(model.client, cap_messages, _CapProbe, probe_timeout),
                    timeout=probe_timeout,
                )
                return combo_key, time.monotonic() - started, "capable", "json-mode"
            except asyncio.TimeoutError:
                return combo_key, time.monotonic() - started, "timeout", None
            except Exception as rescue_exc:
                if kind(rescue_exc) == "transient":
                    return combo_key, time.monotonic() - started, "transient", str(rescue_exc)[:100]
                return combo_key, time.monotonic() - started, "incapable", str(rescue_exc)[:100]

    logger.info("🔬 Capability-probing %d (provider, model) combos (≤%.0fs)...", len(reps), timeout)
    results = await asyncio.gather(*[probe(key, model) for key, model in reps.items()])

    dead_combos: set = set()
    incapable_combos: set = set()
    timed_out_instances: set[str] = set()
    for combo_key, elapsed, status, detail in results:
        provider, base, pipeline = combo_key
        if status == "capable":
            logger.info(
                "✅ Capable: %s/%s (%s) — %.1fs%s",
                provider,
                base,
                pipeline,
                elapsed,
                " [json-mode]" if detail == "json-mode" else "",
            )
            latency_targets = (
                [reps[combo_key]]
                if provider == "google"
                else [model for model in text_pipeline + vision_pipeline if combo_of(model) == combo_key]
            )
            for model in latency_targets:
                if combo_of(model) == combo_key:
                    health.record_success(model.name, latency=elapsed)
            if detail == "json-mode":
                health.force_json_mode(reps[combo_key].name)
        elif status == "dead":
            for model in text_pipeline + vision_pipeline:
                if combo_of(model) == combo_key:
                    health.record_hard_dead(model.name, "MODEL_HARD_DEAD")
            dead_combos.add(combo_key)
            logger.warning("💀 Prune DEAD: %s/%s (%s) — %s", provider, base, pipeline, detail)
        elif status == "incapable":
            incapable_combos.add(combo_key)
            logger.warning("🚫 Prune INCAPABLE (can't structure): %s/%s (%s) — %s", provider, base, pipeline, detail)
        elif status == "timeout":
            probe_timeout = vision_timeout if pipeline == "vision" and vision_timeout is not None else timeout
            logger.warning(
                "🐢 Startup probe timeout: %s/%s (%s) — timed out at %.0fs; retaining in chain",
                provider,
                base,
                pipeline,
                probe_timeout,
            )
            timeout_targets = (
                [reps[combo_key]]
                if provider == "google"
                else [model for model in text_pipeline + vision_pipeline if combo_of(model) == combo_key]
            )
            for model in timeout_targets:
                health.record_failure(model.name, latency=probe_timeout, reliability_failure=False)
                timed_out_instances.add(model.name)
        else:
            logger.info("ℹ️ Transient (kept): %s/%s (%s) — %s", provider, base, pipeline, detail)

    if dead_combos or incapable_combos:
        before_text, before_vision = len(text_pipeline), len(vision_pipeline)
        text_pipeline = apply_capability_gate(text_pipeline, dead_combos, incapable_combos, "TEXT")
        vision_pipeline = apply_capability_gate(vision_pipeline, dead_combos, incapable_combos, "VISION")
        logger.info(
            "🔬 Capability gate — %d dead, %d incapable | TEXT %d→%d, VISION %d→%d",
            len(dead_combos),
            len(incapable_combos),
            before_text,
            len(text_pipeline),
            before_vision,
            len(vision_pipeline),
        )

    if timed_out_instances:
        logger.warning(
            "🐢 Startup probe timed out for %d instance(s); retaining them without persistent failure penalties",
            len(timed_out_instances),
        )
    elif not (dead_combos or incapable_combos):
        logger.info("🔬 Capability probe complete — all %d combos capable", len(reps))

    return text_pipeline, vision_pipeline
