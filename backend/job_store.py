from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    query_type TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'queued',
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    output_dir TEXT,
                    result_json_path TEXT,
                    stdout_log_path TEXT,
                    stderr_log_path TEXT,
                    traceback_log_path TEXT,
                    error TEXT,
                    error_summary TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    retry_of TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "traceback_log_path" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN traceback_log_path TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_cache ON jobs(cache_key, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at)")
            conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        for key in ("cancel_requested", "cache_hit"):
            data[key] = bool(data.get(key))
        return data

    def create_job(
        self,
        *,
        job_id: str,
        query_type: str,
        input_text: str,
        input_hash: str,
        cache_key: str,
        retry_of: str | None = None,
        status: str = "queued",
        stage: str = "queued",
        progress: int = 0,
        message: str = "等待执行",
        cache_hit: bool = False,
        output_dir: str | None = None,
        result_json_path: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, query_type, input_text, input_hash, cache_key, status,
                    stage, progress, message, output_dir, result_json_path,
                    cache_hit, retry_of, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    query_type,
                    input_text,
                    input_hash,
                    cache_key,
                    status,
                    stage,
                    int(progress),
                    message,
                    output_dir,
                    result_json_path,
                    int(cache_hit),
                    retry_of,
                    now,
                    now,
                ),
            )
            conn.commit()
        job = self.get_job(job_id)
        assert job is not None
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row)

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        if not fields:
            return self.get_job(job_id)
        fields["updated_at"] = utc_now_iso()
        normalized: dict[str, Any] = {}
        for key, value in fields.items():
            if isinstance(value, bool):
                normalized[key] = int(value)
            else:
                normalized[key] = value
        assignments = ", ".join(f"{key} = ?" for key in normalized)
        values = list(normalized.values()) + [job_id]
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE job_id = ?", values)
            conn.commit()
        return self.get_job(job_id)

    def request_cancel(self, job_id: str) -> dict[str, Any] | None:
        return self.update_job(job_id, cancel_requested=True, message="用户请求取消任务")

    def find_completed_cache(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE cache_key = ? AND status = 'succeeded' AND result_json_path IS NOT NULL
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (cache_key,),
            ).fetchone()
        job = self._row_to_dict(row)
        if not job:
            return None
        result_path = job.get("result_json_path")
        if result_path and Path(result_path).exists():
            return job
        return None

    def public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "query_type": job["query_type"],
            "input_hash": job["input_hash"],
            "status": job["status"],
            "stage": job["stage"],
            "progress": int(job["progress"] or 0),
            "message": job.get("message") or "",
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "output_dir": job.get("output_dir"),
            "stdout_log_path": job.get("stdout_log_path"),
            "stderr_log_path": job.get("stderr_log_path"),
            "traceback_log_path": job.get("traceback_log_path"),
            "error_summary": job.get("error_summary"),
            "cache_hit": bool(job.get("cache_hit")),
            "retry_of": job.get("retry_of"),
            "result_ready": bool(job.get("result_json_path") and Path(job["result_json_path"]).exists()),
        }
