#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
./agent.sh "Go to https://todomvc.com/examples/react/dist/.

TASK:
1. Add three separate to-do items one by one exactly as follows: 'Learn Rust', 'Master AI Agents', and 'Build Skynet'.
2. Once all three are added, delete ONLY the 'Master AI Agents' task.
3. Verify 'Learn Rust' and 'Build Skynet' remain visible and 'Master AI Agents' is gone from the DOM.
4. If you get confused by identical delete buttons, halt and report."
