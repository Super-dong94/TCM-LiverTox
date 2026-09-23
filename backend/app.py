from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_settings
from .job_manager import JobManager
from .job_store import JobStore
from .predictor import HepatotoxicityPredictor, PredictionError, get_resource_root
from .security import SecurityGuard


ROOT_DIR = get_resource_root(Path(__file__).resolve().parents[1])
FRONTEND_HTML = ROOT_DIR / "frontend" / "Hepatotoxicity_Platform.html"
settings = load_settings(ROOT_DIR)
os.environ.setdefault("TOXHERB_OUTPUT_DIR", str(settings.output_dir))
os.environ.setdefault("TOXHERB_SCRIPT_TIMEOUT_SECONDS", str(settings.script_timeout_seconds))

app = FastAPI(
    title="中药多维-多层级肝毒性预测系统",
    version="2.0.0",
    description="Local FastAPI backend for formula, herb and compound hepatotoxicity prediction.",
)

cors_kwargs: dict[str, Any] = {
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if settings.run_mode == "local":
    cors_kwargs["allow_origins"] = settings.cors_allow_origins or []
    cors_kwargs["allow_origin_regex"] = settings.cors_allow_origin_regex
else:
    cors_kwargs["allow_origins"] = settings.cors_allow_origins
app.add_middleware(CORSMiddleware, **cors_kwargs)
VENDOR_DIR = ROOT_DIR / "frontend" / "vendor"
if VENDOR_DIR.exists():
    app.mount("/vendor", StaticFiles(directory=VENDOR_DIR), name="vendor")

predictor = HepatotoxicityPredictor(ROOT_DIR)
job_store = JobStore(settings.job_db)
job_manager = JobManager(predictor, job_store, settings)
security_guard = SecurityGuard(settings)


def require_api_access(request: Request) -> None:
    security_guard.require_api_access(request)


class PredictionRequest(BaseModel):
    text: str = Field("", description="用户输入的方剂、中药、CID、SMILES 或化学成分名称")
    items: list[str] | None = Field(default=None, description="可选的结构化输入列表")

    def payload(self) -> str | list[str]:
        return self.items if self.items else self.text


class KnowledgeRequest(BaseModel):
    type: Literal["class", "target", "pathway", "go"]
    text: str = ""
    items: list[str] | None = None

    def payload(self) -> str | list[str]:
        return self.items if self.items else self.text


def _handle_prediction_error(exc: PredictionError) -> None:
    raise exc


def _error_payload(code: str, message: str, suggestion: str = "", field: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": code,
        "message": message.splitlines()[0] if message else "请求处理失败。",
        "suggestion": suggestion or "请检查输入内容、运行环境和后台日志。",
        "field": field,
    }
    return payload


@app.exception_handler(PredictionError)
async def prediction_error_handler(_: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": _error_payload("prediction_error", str(exc))},
    )


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "message" in exc.detail:
        detail = exc.detail
    else:
        detail = _error_payload("http_error", str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.get("/")
def index() -> FileResponse:
    if not FRONTEND_HTML.exists():
        raise HTTPException(status_code=500, detail=f"前端文件不存在: {FRONTEND_HTML}")
    return FileResponse(FRONTEND_HTML)


@app.get("/health")
def health() -> dict[str, Any]:
    data = predictor.health()
    data["settings"] = {
        "run_mode": settings.run_mode,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "script_timeout_seconds": settings.script_timeout_seconds,
        "output_dir": str(settings.output_dir),
        "visualization_enabled": settings.enable_visualization,
        "auth_enabled": settings.auth_enabled,
    }
    return data

@app.get("/api/database/stats", dependencies=[Depends(require_api_access)])
def database_stats() -> dict[str, Any]:
    try:
        return predictor.database_stats()
    except PredictionError as exc:
        _handle_prediction_error(exc)

@app.post("/api/predict/formula", dependencies=[Depends(require_api_access)])
def predict_formula(request: PredictionRequest) -> dict[str, Any]:
    try:
        return predictor.predict_formula(request.payload())
    except PredictionError as exc:
        _handle_prediction_error(exc)


@app.post("/api/predict/herb", dependencies=[Depends(require_api_access)])
def predict_herb(request: PredictionRequest) -> dict[str, Any]:
    try:
        return predictor.predict_herb(request.payload())
    except PredictionError as exc:
        _handle_prediction_error(exc)


@app.post("/api/predict/compound", dependencies=[Depends(require_api_access)])
def predict_compound(request: PredictionRequest) -> dict[str, Any]:
    try:
        return predictor.predict_compound(request.payload())
    except PredictionError as exc:
        _handle_prediction_error(exc)


@app.post("/api/search/knowledge", dependencies=[Depends(require_api_access)])
def search_knowledge(request: KnowledgeRequest) -> dict[str, Any]:
    try:
        return predictor.search_knowledge(request.type, request.payload())
    except PredictionError as exc:
        _handle_prediction_error(exc)


@app.post("/api/jobs/predict/{query_type}", dependencies=[Depends(require_api_access)])
def create_prediction_job(query_type: Literal["formula", "herb", "compound"], request: PredictionRequest) -> dict[str, Any]:
    return job_manager.create_prediction_job(query_type, request.payload())


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_api_access)])
def get_job(job_id: str) -> dict[str, Any]:
    return job_manager.get_job(job_id)


@app.get("/api/jobs/{job_id}/result", dependencies=[Depends(require_api_access)])
def get_job_result(job_id: str) -> dict[str, Any]:
    return job_manager.get_result(job_id)


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_api_access)])
def cancel_job(job_id: str) -> dict[str, Any]:
    return job_manager.cancel_job(job_id)


@app.post("/api/jobs/{job_id}/retry", dependencies=[Depends(require_api_access)])
def retry_job(job_id: str) -> dict[str, Any]:
    return job_manager.retry_job(job_id)


@app.get("/api/jobs/{job_id}/sections/{section}", dependencies=[Depends(require_api_access)])
def get_job_section(
    job_id: str,
    section: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort: str | None = None,
    order: Literal["asc", "desc"] = "desc",
    keyword: str | None = None,
) -> dict[str, Any]:
    return job_manager.get_section_page(
        job_id,
        section,
        page=page,
        page_size=page_size,
        sort=sort,
        order=order,
        keyword=keyword,
    )


@app.get("/api/jobs/{job_id}/export", dependencies=[Depends(require_api_access)])
def export_job(job_id: str) -> StreamingResponse:
    return StreamingResponse(job_manager.export_csv(job_id), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="ToxHERB_{job_id}_all.csv"'})


@app.get("/api/jobs/{job_id}/sections/{section}/export", dependencies=[Depends(require_api_access)])
def export_job_section(job_id: str, section: str) -> StreamingResponse:
    stream = job_manager.export_csv(job_id, section)
    return StreamingResponse(stream, media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="ToxHERB_{job_id}_section.csv"'})


@app.get("/api/visual/database", dependencies=[Depends(require_api_access)])
def visual_database() -> dict[str, Any]:
    if not settings.enable_visualization:
        raise PredictionError("可视化接口已关闭。", status_code=503)
    return predictor.build_database_visuals()


@app.get("/api/visual/jobs/{job_id}", dependencies=[Depends(require_api_access)])
def visual_job(job_id: str) -> dict[str, Any]:
    if not settings.enable_visualization:
        raise PredictionError("可视化接口已关闭。", status_code=503)
    result = job_manager.get_result(job_id)
    return {"ok": True, "job_id": job_id, "chart_specs": result.get("chart_specs", [])}


@app.post("/api/visual/knowledge", dependencies=[Depends(require_api_access)])
def visual_knowledge(request: KnowledgeRequest) -> dict[str, Any]:
    if not settings.enable_visualization:
        raise PredictionError("可视化接口已关闭。", status_code=503)
    result = predictor.search_knowledge(request.type, request.payload())
    return {
        "ok": True,
        "query_type": request.type,
        "summary": result.get("summary", {}),
        "chart_specs": result.get("chart_specs", []),
    }
