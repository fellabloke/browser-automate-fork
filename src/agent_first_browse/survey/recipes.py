"""Verified, page-local survey recipe memory.

Recipes are exact-page accelerators, not broad scripts. They store semantic
targets rather than snapshot IDs, require two verified successes before replay,
and are disabled automatically when their success rate drops.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from agent_first_browse.survey.context import (
    _element_label,
    canonical_survey_url,
    is_image_code_page,
    normalized_survey_question_text,
    survey_gate_violation,
    survey_validation_evidence,
)


RECIPE_VERIFIER_VERSION = 3


def _enabled() -> bool:
    return os.getenv("SURVEY_RECIPE_MEMORY_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off", "",
    }


def survey_page_recipe_signature(
    url: str, page_text: str, selector_map: dict[str, dict]
) -> str:
    """Exact semantic signature: stable route + normalized prompt + controls."""
    text = normalized_survey_question_text(page_text)[:3500]
    controls = []
    for element in (selector_map or {}).values():
        label = re.sub(
            r"\s*\[(?:selected|checked|chosen|disabled|empty|filled:.*?)\]\s*",
            " ", _element_label(element), flags=re.IGNORECASE,
        ).strip()
        controls.append(
            "|".join((
                str(element.get("kind") or "").lower(),
                str(element.get("control_type") or "").lower(),
                label[:180],
                str(element.get("choice_group") or "")[:80],
            ))
        )
    material = "\n".join((canonical_survey_url(url), text, *sorted(controls)))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32] if text else ""


class SurveyRecipeMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(
            path or os.getenv("SURVEY_RECIPE_DB", "persistence/survey_recipes.db")
        )

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=3.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS survey_recipes (
                signature TEXT NOT NULL,
                action_key TEXT NOT NULL,
                route TEXT NOT NULL,
                recipe_json TEXT NOT NULL,
                successes INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0,
                avg_ms REAL NOT NULL DEFAULT 0,
                last_used REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (signature, action_key)
            )
            """
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS survey_recipe_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = connection.execute(
            "SELECT value FROM survey_recipe_meta WHERE key='verifier_version'"
        ).fetchone()
        try:
            version = int(row[0]) if row else 0
        except (TypeError, ValueError):
            version = 0
        if version < RECIPE_VERIFIER_VERSION:
            # Legacy counters were awarded for any DOM/question-text churn,
            # including required-field banners and autocomplete menus. Retain
            # recipes for audit, but force every action to earn trust again
            # under stable-question, action-aware transition verification.
            connection.execute(
                "UPDATE survey_recipes SET successes=0,failures=0,last_used=0"
            )
            connection.execute(
                "INSERT INTO survey_recipe_meta(key,value) VALUES('verifier_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(RECIPE_VERIFIER_VERSION),),
            )
            connection.commit()
        return connection

    @staticmethod
    def _recipe_for(action: dict[str, Any], selector_map: dict[str, dict]) -> dict[str, Any] | None:
        verb = str(action.get("verb") or "")
        basis = str(action.get("answer_basis") or "").lower()
        if verb not in {"click", "type", "select_option"}:
            return None
        if basis not in {
            "page_navigation", "configured_profile_fact", "attention_instruction",
            "objective_reasoning",
        }:
            return None
        element = selector_map.get(str(action.get("element_id") or "")) or {}
        target_label = _element_label(element)
        if not target_label:
            return None
        recipe = {
            "verb": verb,
            "target_label": target_label[:220],
            "control_type": str(element.get("control_type") or "").lower(),
            "answer_basis": basis,
            "question_text": str(action.get("question_text") or "")[:300],
            "profile_update_category": str(action.get("profile_update_category") or "none")[:40],
            "profile_update_key": str(action.get("profile_update_key") or "")[:80],
        }
        if verb == "select_option":
            # Dropdown values are safe only when they originate from a configured
            # profile fact. Literal subjective values are deliberately not stored.
            if basis != "configured_profile_fact":
                return None
            recipe["option"] = str(action.get("text") or "")[:120]
        return recipe

    def observe_success(
        self,
        *,
        url: str,
        page_text: str,
        selector_map: dict[str, dict],
        action: dict[str, Any],
        elapsed_ms: float = 0.0,
        verified_transition: bool = False,
    ) -> None:
        if (
            not _enabled()
            or not verified_transition
            or is_image_code_page(page_text)
            or survey_validation_evidence(page_text)
        ):
            return
        signature = survey_page_recipe_signature(url, page_text, selector_map)
        recipe = self._recipe_for(action, selector_map)
        if not signature or not recipe:
            return
        action_key = hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT successes, avg_ms FROM survey_recipes WHERE signature=? AND action_key=?",
                (signature, action_key),
            ).fetchone()
            old_successes, old_avg = row if row else (0, 0.0)
            successes = int(old_successes) + 1
            avg_ms = (
                ((float(old_avg) * int(old_successes)) + max(0.0, float(elapsed_ms))) / successes
            )
            connection.execute(
                """
                INSERT INTO survey_recipes
                    (signature, action_key, route, recipe_json, successes, failures, avg_ms, last_used)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(signature, action_key) DO UPDATE SET
                    recipe_json=excluded.recipe_json,
                    successes=excluded.successes,
                    avg_ms=excluded.avg_ms,
                    last_used=excluded.last_used
                """,
                (
                    signature, action_key, canonical_survey_url(url),
                    json.dumps(recipe, separators=(",", ":")), successes, avg_ms, now,
                ),
            )

    def observe_failure(
        self,
        *,
        url: str,
        page_text: str,
        selector_map: dict[str, dict],
        action: dict[str, Any],
    ) -> None:
        """Record a forward/replayed action that failed to leave its question."""
        if not _enabled() or is_image_code_page(page_text):
            return
        signature = survey_page_recipe_signature(url, page_text, selector_map)
        recipe = self._recipe_for(action, selector_map)
        if not signature or not recipe:
            return
        action_key = hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO survey_recipes
                    (signature, action_key, route, recipe_json, successes, failures, avg_ms, last_used)
                VALUES (?, ?, ?, ?, 0, 1, 0, ?)
                ON CONFLICT(signature, action_key) DO UPDATE SET
                    failures=survey_recipes.failures+1,
                    last_used=excluded.last_used
                """,
                (
                    signature, action_key, canonical_survey_url(url),
                    json.dumps(recipe, separators=(",", ":")), time.time(),
                ),
            )

    def record_replay_failure(self, signature: str, action_key: str) -> None:
        if not signature or not action_key:
            return
        with self._connect() as connection:
            connection.execute(
                "UPDATE survey_recipes SET failures=failures+1,last_used=? "
                "WHERE signature=? AND action_key=?",
                (time.time(), signature, action_key),
            )

    def recall(
        self,
        *,
        url: str,
        page_text: str,
        selector_map: dict[str, dict],
        profile: dict[str, Any],
    ) -> dict[str, Any] | None:
        if (
            not _enabled()
            or is_image_code_page(page_text)
            or survey_validation_evidence(page_text)
        ):
            return None
        signature = survey_page_recipe_signature(url, page_text, selector_map)
        if not signature:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT action_key,recipe_json,successes,failures FROM survey_recipes "
                "WHERE signature=? ORDER BY successes DESC,last_used DESC",
                (signature,),
            ).fetchall()
        for action_key, raw, successes, failures in rows:
            total = int(successes) + int(failures)
            if int(successes) < 2 or total <= 0 or int(successes) / total < 0.9:
                continue
            try:
                recipe = json.loads(raw)
            except (TypeError, ValueError):
                continue
            matches = [
                (str(element_id), element)
                for element_id, element in selector_map.items()
                if _element_label(element) == str(recipe.get("target_label") or "")
                and (
                    not recipe.get("control_type")
                    or str(element.get("control_type") or "").lower() == recipe["control_type"]
                )
            ]
            if len(matches) != 1:
                continue
            element_id, element = matches[0]
            action = {
                "verb": recipe["verb"], "element_id": element_id,
                "target_name": _element_label(element)[:160],
                "target_context": str(element.get("hint") or "")[:120],
                "text": recipe.get("option"), "url": None, "x": None, "y": None,
                "question_text": recipe.get("question_text") or page_text[:300],
                "answer_basis": recipe.get("answer_basis") or "page_navigation",
                "profile_update_category": recipe.get("profile_update_category") or "none",
                "profile_update_key": recipe.get("profile_update_key") or "",
                "profile_update_mode": "set" if recipe.get("profile_update_key") else "none",
                "profile_update_value": "", "profile_update_reason": "Verified recipe replay.",
                "rationale": "High-confidence exact-page recipe replay.",
                "reasoning": "This semantic page/action has succeeded at least twice.",
                "expected_change": "The same verified local state transition occurs.",
                "risk_level": "REVERSIBLE", "reversible": True,
                "recipe_signature": signature, "recipe_action_key": action_key,
                "queued_actions": [], "execution_mode": "single_action",
            }
            if action["verb"] == "type":
                try:
                    from survey_profile import enforce_typed_profile_fact
                    action, note, violation = enforce_typed_profile_fact(
                        action, profile, selector_map, page_text=page_text
                    )
                    if violation or not note or not action.get("text"):
                        continue
                except Exception:
                    continue
            if not survey_gate_violation(
                action, selector_map, page_text=page_text, continuous_mode=True
            ):
                return action
        return None


_MEMORY: SurveyRecipeMemory | None = None


def get_survey_recipe_memory() -> SurveyRecipeMemory:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = SurveyRecipeMemory()
    return _MEMORY
