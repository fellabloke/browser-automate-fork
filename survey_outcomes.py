"""Persistent provider outcome learning and per-cycle survey telemetry."""

from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DEFAULT_PATH = Path(__file__).parent / "persistence" / "survey_outcomes.json"


def _configured_path() -> Path:
    raw = str(os.getenv("SURVEY_OUTCOME_PATH", "")).strip()
    return Path(raw).expanduser() if raw else DEFAULT_PATH


def provider_key(url: str) -> str:
    try:
        parsed = urlsplit(str(url or "").strip())
        path = (parsed.path or "/").rstrip("/") or "/"
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))
    except Exception:
        return str(url or "").split("?", 1)[0].rstrip("/")


def provider_host(url: str) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower()
    except Exception:
        return ""


class SurveyOutcomeStore:
    """Learn expected completions/hour and retain auditable cycle records."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _configured_path()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "providers": {}, "cycles": []}
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(document, dict):
                document.setdefault("providers", {})
                document.setdefault("cycles", [])
                return document
        except (OSError, ValueError, TypeError):
            pass
        return {"version": 1, "providers": {}, "cycles": []}

    def _write(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def record(
        self,
        *,
        panel_url: str,
        result: str,
        elapsed_seconds: float,
        survey_url: str = "",
        questions: int = 0,
        reward: str = "",
        offer_minutes: float | None = None,
    ) -> dict[str, Any]:
        key = provider_key(panel_url)
        if not key:
            return {}
        normalized = str(result or "abandoned").lower()
        completed = normalized.startswith("completed")
        screened_out = normalized.startswith("screened_out")
        elapsed = max(1.0, min(float(elapsed_seconds or 0.0), 24 * 3600.0))
        document = self.load()
        providers = document.setdefault("providers", {})
        stats = providers.setdefault(key, {
            "attempts": 0,
            "completions": 0,
            "screened_out": 0,
            "abandoned": 0,
            "total_seconds": 0.0,
            "last_result": "",
            "last_updated": 0.0,
        })
        stats["attempts"] = int(stats.get("attempts", 0) or 0) + 1
        stats["completions"] = int(stats.get("completions", 0) or 0) + int(completed)
        stats["screened_out"] = int(stats.get("screened_out", 0) or 0) + int(screened_out)
        stats["abandoned"] = int(stats.get("abandoned", 0) or 0) + int(
            not completed and not screened_out
        )
        stats["total_seconds"] = round(float(stats.get("total_seconds", 0.0) or 0.0) + elapsed, 3)
        stats["last_result"] = normalized[:120]
        stats["last_updated"] = time.time()

        cycle = {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "panel_url": key,
            "panel_host": provider_host(panel_url),
            "survey_host": provider_host(survey_url),
            "result": normalized[:120],
            "elapsed_seconds": round(elapsed, 3),
            "questions": max(0, int(questions or 0)),
            "reward": str(reward or "")[:80],
            "offer_minutes": (
                round(float(offer_minutes), 2) if offer_minutes is not None else None
            ),
        }
        document["cycles"] = (list(document.get("cycles") or []) + [cycle])[-500:]
        document["updated_at"] = time.time()
        self._write(document)
        return cycle

    @staticmethod
    def expected_completions_per_hour(stats: dict[str, Any]) -> float:
        attempts = max(0, int(stats.get("attempts", 0) or 0))
        completions = max(0, int(stats.get("completions", 0) or 0))
        # Conservative Beta(1,2) qualification+completion prior and two
        # 15-minute pseudo-observations keep one lucky short completion from
        # monopolising the run while still learning quickly.
        probability = (completions + 1.0) / (attempts + 3.0)
        mean_seconds = (
            float(stats.get("total_seconds", 0.0) or 0.0) + 1800.0
        ) / (attempts + 2.0)
        mean_seconds = max(60.0, mean_seconds)
        exploration = 0.20 / math.sqrt(attempts + 1.0)
        return probability * 3600.0 / mean_seconds + exploration

    def choose_index(
        self,
        urls: list[str],
        *,
        current_index: int | None = None,
        pending_failure: bool = False,
        exclude_current: bool = False,
    ) -> int:
        if not urls:
            return 0
        document = self.load()
        providers = document.get("providers") or {}
        candidates: list[tuple[float, int]] = []
        for index, url in enumerate(urls):
            if exclude_current and current_index == index and len(urls) > 1:
                continue
            stats = deepcopy(providers.get(provider_key(url)) or {})
            if pending_failure and current_index == index:
                stats["attempts"] = int(stats.get("attempts", 0) or 0) + 1
                stats["total_seconds"] = float(stats.get("total_seconds", 0.0) or 0.0) + 900.0
            candidates.append((self.expected_completions_per_hour(stats), index))
        # Stable list order resolves an unexplored tie.
        return max(candidates, key=lambda item: (item[0], -item[1]))[1]


_STORE: SurveyOutcomeStore | None = None


def get_survey_outcome_store() -> SurveyOutcomeStore:
    global _STORE
    if _STORE is None:
        _STORE = SurveyOutcomeStore()
    return _STORE


def choose_survey_provider_index(
    urls: list[str],
    *,
    current_index: int | None = None,
    pending_failure: bool = False,
    exclude_current: bool = False,
) -> int:
    return get_survey_outcome_store().choose_index(
        urls,
        current_index=current_index,
        pending_failure=pending_failure,
        exclude_current=exclude_current,
    )
