from __future__ import annotations

import json
import shutil
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .job_store import JobStore
from .predictor import HepatotoxicityPredictor, PredictionError, split_items


ProgressCallback = Callable[[str, int, str], None]


class JobManager:
    def __init__(self, predictor: HepatotoxicityPredictor, store: JobStore, settings: Settings) -> None:
        self.predictor = predictor
        self.store = store
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="toxherb-job")
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def create_prediction_job(self, query_type: str, payload: str | list[str], retry_of: str | None = None) -> dict[str, Any]:
        if query_type not in {"formula", "herb", "compound"}:
            raise PredictionError(f"不支持的预测类型: {query_type}", status_code=404)
        items = split_items(payload)
        if not items:
            raise PredictionError("请输入至少一个检索目标。")

        input_text = "\n".join(items)
        input_hash = self.predictor.input_hash(query_type, items)
        cache_key = self.predictor.prediction_cache_key(query_type, items)
        job_id = uuid.uuid4().hex

        cached = self.store.find_completed_cache(cache_key)
        if cached and not retry_of:
            job = self.store.create_job(
                job_id=job_id,
                query_type=query_type,
                input_text=input_text,
                input_hash=input_hash,
                cache_key=cache_key,
                retry_of=None,
                status="succeeded",
                stage="cache_hit",
                progress=100,
                message="命中历史缓存，已复用预测结果。",
                cache_hit=True,
                output_dir=cached.get("output_dir"),
                result_json_path=cached.get("result_json_path"),
            )
            return self.store.public_job(job)

        job = self.store.create_job(
            job_id=job_id,
            query_type=query_type,
            input_text=input_text,
            input_hash=input_hash,
            cache_key=cache_key,
            retry_of=retry_of,
        )
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = cancel_event
            self._futures[job_id] = self.executor.submit(self._run_prediction_job, job_id, query_type, items, cancel_event)
        return self.store.public_job(job)

    def get_job(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise PredictionError("任务不存在。", status_code=404)
        return self.store.public_job(job)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise PredictionError("任务不存在。", status_code=404)
        self.store.request_cancel(job_id)
        with self._lock:
            event = self._cancel_events.get(job_id)
            if event:
                event.set()
            future = self._futures.get(job_id)
            if future and future.cancel():
                self.store.update_job(job_id, status="cancelled", stage="cancelled", progress=100, message="任务已取消。")
        updated = self.store.get_job(job_id)
        assert updated is not None
        return self.store.public_job(updated)

    def retry_job(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise PredictionError("任务不存在。", status_code=404)
        return self.create_prediction_job(job["query_type"], job["input_text"], retry_of=job_id)

    def get_result(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise PredictionError("任务不存在。", status_code=404)
        if job["status"] != "succeeded":
            raise PredictionError("任务尚未完成。", status_code=409)
        result_path = job.get("result_json_path")
        if not result_path or not Path(result_path).exists():
            raise PredictionError("任务结果文件不存在。", status_code=500)
        return json.loads(Path(result_path).read_text(encoding="utf-8"))

    def get_section_page(
        self,
        job_id: str,
        section: str,
        *,
        page: int,
        page_size: int,
        sort: str | None = None,
        order: str = "desc",
        keyword: str | None = None,
    ) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if not job:
            raise PredictionError("任务不存在。", status_code=404)
        if job["status"] != "succeeded":
            raise PredictionError("任务尚未完成，暂不能读取分页结果。", status_code=409)
        output_dir = job.get("output_dir")
        if not output_dir:
            raise PredictionError("任务输出目录不存在。", status_code=500)
        return self.predictor.get_output_section_page(
            query_type=job["query_type"],
            output_dir=Path(output_dir),
            section_id=section,
            page=page,
            page_size=page_size,
            sort=sort,
            order=order,
            keyword=keyword,
            job_id=job_id,
        )

    def _run_prediction_job(
        self,
        job_id: str,
        query_type: str,
        items: list[str],
        cancel_event: threading.Event,
    ) -> None:
        def progress(stage: str, percent: int, message: str) -> None:
            if cancel_event.is_set():
                return
            self.store.update_job(
                job_id,
                stage=stage,
                progress=max(0, min(99, int(percent))),
                message=message,
                status="running",
            )

        run_meta: dict[str, Any] = {}
        self.store.update_job(job_id, status="running", stage="preparing", progress=1, message="正在解析输入实体。")
        try:
            result, run_meta = self.predictor.run_prediction_pipeline(
                query_type,
                items,
                job_id=job_id,
                progress_callback=progress,
                cancel_event=cancel_event,
            )
            if cancel_event.is_set():
                self.store.update_job(job_id, status="cancelled", stage="cancelled", progress=100, message="任务已取消。")
                return

            result_dir = Path(run_meta["output_dir"])
            result_path = result_dir / "toxherb_result.json"
            result["job_id"] = job_id
            result["input_hash"] = self.predictor.input_hash(query_type, items)
            result["cache_key"] = self.predictor.prediction_cache_key(query_type, items)
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.store.update_job(
                job_id,
                status="succeeded",
                stage="complete",
                progress=100,
                message="预测完成。",
                output_dir=str(result_dir),
                result_json_path=str(result_path),
                stdout_log_path=run_meta.get("stdout_log_path"),
                stderr_log_path=run_meta.get("stderr_log_path"),
            )
        except PredictionError as exc:
            if cancel_event.is_set() or exc.status_code == 499:
                self.store.update_job(job_id, status="cancelled", stage="cancelled", progress=100, message="任务已取消。")
                return
            self.store.update_job(
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                message="预测失败。",
                error=str(exc),
                error_summary=str(exc).splitlines()[0][:500],
                output_dir=getattr(exc, "output_dir", run_meta.get("output_dir")),
                stdout_log_path=getattr(exc, "stdout_log_path", None),
                stderr_log_path=getattr(exc, "stderr_log_path", None),
                traceback_log_path=getattr(exc, "traceback_log_path", None),
            )
        except Exception as exc:  # noqa: BLE001 - keep worker failures visible through API
            log_dir = self.settings.job_log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job_id}.traceback.log"
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            self.store.update_job(
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                message="预测失败。",
                error=str(exc),
                error_summary=str(exc).splitlines()[0][:500],
                output_dir=getattr(exc, "output_dir", run_meta.get("output_dir")),
                stdout_log_path=getattr(exc, "stdout_log_path", run_meta.get("stdout_log_path")),
                stderr_log_path=getattr(exc, "stderr_log_path", run_meta.get("stderr_log_path")),
                traceback_log_path=str(log_path),
            )
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._futures.pop(job_id, None)


def copy_cached_result(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
