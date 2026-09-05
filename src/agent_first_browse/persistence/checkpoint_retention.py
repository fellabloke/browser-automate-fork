"""Bound SQLite checkpoint growth for long-running graph sessions."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def checkpoint_retention_enabled() -> bool:
    return os.getenv("CHECKPOINT_RETENTION_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off", "",
    }


def prune_checkpoint_database(
    database_path: str | Path,
    active_thread_id: str,
    *,
    keep_active: int | None = None,
    keep_prior_threads: int | None = None,
    keep_per_prior_thread: int | None = None,
) -> dict[str, int]:
    """Keep recent resumability while deleting redundant historical snapshots.

    Each LangGraph checkpoint is a complete state snapshot, so retaining a recent
    window is sufficient for crash recovery. Survey profiles and run logs live in
    separate files and are never touched here.
    """
    if not checkpoint_retention_enabled():
        return {"checkpoints_deleted": 0, "threads_deleted": 0}

    keep_active = keep_active or _env_int("CHECKPOINT_KEEP_ACTIVE", 80, 10)
    keep_prior_threads = keep_prior_threads or _env_int(
        "CHECKPOINT_KEEP_PRIOR_THREADS", 12, 1
    )
    keep_per_prior_thread = keep_per_prior_thread or _env_int(
        "CHECKPOINT_KEEP_PER_PRIOR_THREAD", 2, 1
    )

    checkpoints_deleted = 0
    threads_deleted = 0
    with sqlite3.connect(str(database_path), timeout=1.0) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT thread_id, MAX(checkpoint_id) AS newest "
            "FROM checkpoints GROUP BY thread_id ORDER BY newest DESC"
        )
        rows = cur.fetchall()
        prior_threads = [str(row[0]) for row in rows if str(row[0]) != active_thread_id]
        retained = {active_thread_id, *prior_threads[:keep_prior_threads]}

        for thread_id in prior_threads[keep_prior_threads:]:
            cur.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,))
            checkpoints_deleted += int(cur.fetchone()[0] or 0)
            cur.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            cur.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            threads_deleted += 1

        for thread_id in retained:
            keep = keep_active if thread_id == active_thread_id else keep_per_prior_thread
            cur.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? "
                "ORDER BY checkpoint_id DESC LIMIT -1 OFFSET ?",
                (thread_id, keep),
            )
            old_ids = [str(row[0]) for row in cur.fetchall()]
            if not old_ids:
                continue
            checkpoints_deleted += len(old_ids)
            cur.executemany(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_id = ?",
                [(thread_id, checkpoint_id) for checkpoint_id in old_ids],
            )
            cur.executemany(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                [(thread_id, checkpoint_id) for checkpoint_id in old_ids],
            )
        cur.execute(
            "DELETE FROM writes WHERE NOT EXISTS ("
            "SELECT 1 FROM checkpoints c WHERE c.thread_id = writes.thread_id "
            "AND c.checkpoint_ns = writes.checkpoint_ns "
            "AND c.checkpoint_id = writes.checkpoint_id)"
        )
        conn.commit()

    return {
        "checkpoints_deleted": checkpoints_deleted,
        "threads_deleted": threads_deleted,
    }
