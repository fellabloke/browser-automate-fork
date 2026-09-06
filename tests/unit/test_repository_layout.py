"""Keep the repository root small and package-owned."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = {
    ".agents", ".github", "docs", "examples", "scripts", "src", "tests",
    ".env.example", ".gitignore", "AGENTS.md", "ARCHITECTURE.md", "README.md",
    "LICENSE", "pyproject.toml", "agent.sh", "Start-Agent.ps1", "skills-lock.json",
}
LOCAL_ONLY = {
    ".git", ".codex", ".env", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__",
    "agent_first_ide.egg-info", "logs", "persistence",
}


def test_root_contains_only_canonical_or_local_environment_entries():
    entries = {path.name for path in ROOT.iterdir()}
    unexpected = entries - CANONICAL_ROOT - LOCAL_ONLY
    assert not unexpected, f"unexpected repository-root entries: {sorted(unexpected)}"


def test_canonical_source_and_package_data_are_present():
    package = ROOT / "src" / "agent_first_browse"
    assert (package / "__init__.py").is_file()
    profile = package / "survey" / "survey_profiles.example.json"
    assert profile.is_file()
    assert isinstance(json.loads(profile.read_text(encoding="utf-8")), dict)
    assert not (ROOT / "survey_profiles.example.json").exists()


def test_retired_runtime_trees_do_not_reappear():
    for name in ("orchestrator", "python-orchestrator", "skills", "workers"):
        assert not (ROOT / name).exists(), f"retired runtime tree returned: {name}"


def test_root_has_no_unowned_python_or_sample_files():
    assert not list(ROOT.glob("*.py"))
    assert not list(ROOT.glob("*.txt"))
    assert [path.name for path in ROOT.glob("*.json")] == ["skills-lock.json"]
