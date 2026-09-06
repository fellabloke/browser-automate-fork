"""Reset local Gemini/Cloudflare usage parking while an agent may be running.

Usage:
    .venv/bin/python reset_model_usage.py

The running agent notices the health-cache change on its next scheduling pass
and reloads the reset without requiring a restart.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent_first_browse.models import ProviderHealthTracker


project_root = Path(__file__).resolve().parent
health_path = Path(os.getenv("MODEL_HEALTH_PATH", "persistence/model_health.json"))
if not health_path.is_absolute():
    health_path = project_root / health_path

tracker = ProviderHealthTracker(health_path)
count = tracker.reset_usage_limits()
parked = tracker.clear_timeout_parking()
print(f"Reset {count} Gemini/Cloudflare usage ledgers and cleared {parked} timeout parks in {health_path}")
