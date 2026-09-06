#!/usr/bin/env python3
"""Guard canonical repository naming and dependency-direction invariants."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", ".codex", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {
    ".md", ".py", ".sh", ".ps1", ".toml", ".yaml", ".yml", ".ini", ".cfg",
    ".txt", ".json", ".env",
}

# These are external contracts, not project lineage. Keep the exceptions narrow
# and visible so a new internal generation label cannot hide behind this guard.
ALLOWED_TEXT = (
    "GPLv3",
    "schema-v5",
    "pre-v5-backup",
    "LANGCHAIN_TRACING_V2",
    "application/vnd.github.v3+json",
    "https://github.com/SandeepAi369/Agent-first-ide.git",
)
INTERNAL_TOKEN = re.compile(
    r"(?:^|[_-])v(?:8c|(?:1[1-9]|[2-9]\d+))(?=$|[_./-])|\bV(?:\d+)(?:\.\d+)?\b"
)


def _ignored(path: Path) -> bool:
    return bool(set(path.relative_to(ROOT).parts) & IGNORED_PARTS)


def main() -> int:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or _ignored(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if INTERNAL_TOKEN.search(path.name) and not path.name.endswith(".pre-v5-backup"):
            findings.append(f"filename: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scrubbed = text
        for allowed in ALLOWED_TEXT:
            scrubbed = scrubbed.replace(allowed, "")
        for line_number, line in enumerate(scrubbed.splitlines(), 1):
            if INTERNAL_TOKEN.search(line) and path.resolve() != Path(__file__).resolve():
                findings.append(f"text: {relative}:{line_number}: {line.strip()}")

    package_root = ROOT / "src" / "agent_first_browse"
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if "promotion.browser_promoter" in text and "/promotion/" not in relative:
            findings.append(f"core-to-promotion backedge: {relative}")
        if re.search(r"^\s*(?:from|import)\s+(?:app\.|python_orchestrator|orchestrator(?:\.|\s))", text, re.MULTILINE):
            findings.append(f"retired namespace import: {relative}")
        if "sys.path" in text or "PYTHONPATH" in text:
            findings.append(f"path hack in canonical source: {relative}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'name = "agent-first-browse"' not in pyproject:
        findings.append("project identity: pyproject.toml must use agent-first-browse")
    if findings:
        print("repository naming/architecture violations detected:")
        print("\n".join(findings))
        return 1
    print("repository naming audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
