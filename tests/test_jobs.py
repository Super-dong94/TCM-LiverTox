from __future__ import annotations

import time
from pathlib import Path

from backend.config import Settings
from backend.job_manager import JobManager
from backend.job_store import JobStore


class FakePredictor:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls = 0

    def input_hash(self, query_type, items):  # noqa: ANN001
        return f"hash-{query_type}-{'-'.join(items)}"

    def prediction_cache_key(self, query_type, items):  # noqa: ANN001
        return f"cache-{query_type}-{'-'.join(items)}"

    def run_prediction_pipeline(self, query_type, items, *, job_id=None, progress_callback=None, cancel_event=None):  # noqa: ANN001
        self.calls += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            progress_callback("entity_resolution", 10, "解析完成")
            progress_callback("complete", 95, "准备结果")
        return (
            {
                "ok": True,
                "query_type": query_type,
                "query": items,
                "summary": {"raw_max_toxic_probability": 0.8},
                "chart_specs": [],
            },
            {"output_dir": str(self.output_dir), "stdout_log_path": None, "stderr_log_path": None},
        )

    def get_output_section_page(self, **kwargs):  # noqa: ANN003
        return {"ok": True, "section": {"id": kwargs["section_id"]}, "rows": []}


class FailingAfterOutputPredictor(FakePredictor):
    def run_prediction_pipeline(self, query_type, items, *, job_id=None, progress_callback=None, cancel_event=None):  # noqa: ANN001
        self.calls += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = self.output_dir / "worker.stdout.log"
        stderr_log = self.output_dir / "worker.stderr.log"
        stdout_log.write_text("script completed\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        if progress_callback:
            progress_callback("response_building", 92, "构建响应")
        exc = RuntimeError("response build exploded")
        exc.output_dir = str(self.output_dir)
        exc.stdout_log_path = str(stdout_log)
        exc.stderr_log_path = str(stderr_log)
        raise exc


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        root_dir=tmp_path,
        run_mode="local",
        cors_allow_origins=[],
        cors_allow_origin_regex=None,
        max_concurrent_jobs=1,
        script_timeout_seconds=30,
        output_dir=tmp_path / "outputs",
        user_dir=tmp_path,
        job_db=tmp_path / "jobs.sqlite",
        job_log_dir=tmp_path / "logs",
        enable_visualization=True,
        auth_enabled=False,
        api_token=None,
        rate_limit_per_minute=60,
    )


def wait_for_status(manager: JobManager, job_id: str, status: str) -> dict[str, object]:
    for _ in range(50):
        job = manager.get_job(job_id)
        if job["status"] == status:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach {status}")


def test_job_create_complete_and_cache_hit(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    predictor = FakePredictor(tmp_path / "run-output")
    manager = JobManager(predictor, JobStore(settings.job_db), settings)

    created = manager.create_prediction_job("herb", "何首乌")
    done = wait_for_status(manager, created["job_id"], "succeeded")
    assert done["result_ready"] is True
    assert manager.get_result(created["job_id"])["summary"]["raw_max_toxic_probability"] == 0.8

    cached = manager.create_prediction_job("herb", "何首乌")
    assert cached["status"] == "succeeded"
    assert cached["cache_hit"] is True
    assert predictor.calls == 1


def test_job_section_proxy(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    predictor = FakePredictor(tmp_path / "run-output")
    manager = JobManager(predictor, JobStore(settings.job_db), settings)
    created = manager.create_prediction_job("compound", "3220")
    wait_for_status(manager, created["job_id"], "succeeded")
    page = manager.get_section_page(created["job_id"], "toxic_compounds", page=1, page_size=50)
    assert page["section"]["id"] == "toxic_compounds"


def test_failed_job_preserves_output_and_log_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    predictor = FailingAfterOutputPredictor(tmp_path / "run-output")
    manager = JobManager(predictor, JobStore(settings.job_db), settings)

    created = manager.create_prediction_job("formula", "生血宝合剂")
    failed = wait_for_status(manager, created["job_id"], "failed")

    assert failed["output_dir"] == str(tmp_path / "run-output")
    assert failed["stdout_log_path"].endswith("worker.stdout.log")
    assert failed["stderr_log_path"].endswith("worker.stderr.log")
    assert failed["traceback_log_path"]
    assert Path(failed["traceback_log_path"]).exists()
    assert "response build exploded" in failed["error_summary"]
