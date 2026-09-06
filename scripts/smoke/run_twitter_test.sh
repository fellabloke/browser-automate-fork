#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
"${PYTHON_BIN:-$ROOT/.venv/bin/python}" -m agent_first_browse.cli run "Navigate to https://x.com. Find the compose post / 'What is happening?!' box. Write a tweet with a 10-word hook and a 25-word paragraph about how autonomous AI browser agents can now completely bypass anti-bot traps like Amazon using stealth CDP injection. Click the 'Post' button to publish it. Wait for the confirmation that the tweet was sent and verify it is live."
