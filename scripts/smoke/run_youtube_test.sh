#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
"${PYTHON_BIN:-$ROOT/.venv/bin/python}" -m agent_first_browse.cli run "Navigate to YouTube (https://www.youtube.com). Search for 'BR Chopra mahabharat episode 2' using the search bar and press enter. Click on the correct video from the search results to start playing it. Wait to verify the video has successfully started playing, and then mark the task as complete."
