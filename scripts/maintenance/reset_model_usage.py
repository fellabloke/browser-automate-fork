"""Reset local model usage parking while an agent may be running."""

from __future__ import annotations

import os
from pathlib import Path

from agent_first_browse.models import ProviderHealthTracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
health_path = Path(os.getenv("MODEL_HEALTH_PATH", "persistence/model_health.json"))
if not health_path.is_absolute():
    health_path = PROJECT_ROOT / health_path

tracker = ProviderHealthTracker(health_path)
count = tracker.reset_usage_limits()
parked = tracker.clear_timeout_parking()
print(f"Reset {count} usage ledgers and cleared {parked} timeout parks in {health_path}")
