#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

if [[ -z "${RUFF_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/ruff" ]]; then
        RUFF_BIN="$REPO_ROOT/.venv/bin/ruff"
    else
        RUFF_BIN="ruff"
    fi
fi

if ! command -v "$RUFF_BIN" >/dev/null 2>&1 && [[ ! -x "$RUFF_BIN" ]]; then
    echo "error: Ruff is required for the canonical check; install the dev dependencies first" >&2
    exit 127
fi

echo "==> Ruff"
"$RUFF_BIN" check --no-fix --select E9,F63,F7 .

echo "==> Shell syntax"
bash -n "$REPO_ROOT/scripts/check.sh"

echo "==> Deterministic pytest"
"$PYTHON_BIN" -m pytest -q tests/unit tests/regression
