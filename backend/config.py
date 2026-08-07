from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .predictor import get_user_dir


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        return max(minimum, value)
    return value


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    run_mode: str
    cors_allow_origins: list[str]
    cors_allow_origin_regex: str | None
    max_concurrent_jobs: int
    script_timeout_seconds: int
    output_dir: Path
    user_dir: Path
    job_db: Path
    job_log_dir: Path
    enable_visualization: bool
    auth_enabled: bool
    api_token: str | None
    rate_limit_per_minute: int


def load_settings(root_dir: Path) -> Settings:
    run_mode = os.getenv("TOXHERB_RUN_MODE", "local").strip().lower() or "local"
    if run_mode not in {"local", "server"}:
        raise RuntimeError("TOXHERB_RUN_MODE 只能为 local 或 server。")

    user_dir = Path(os.getenv("TOXHERB_USER_DIR", str(get_user_dir()))).resolve()
    output_dir = Path(os.getenv("TOXHERB_OUTPUT_DIR", str(user_dir / "prediction_outputs"))).resolve()
    job_db = Path(os.getenv("TOXHERB_JOB_DB", str(user_dir / "jobs.sqlite"))).resolve()
    job_log_dir = Path(os.getenv("TOXHERB_JOB_LOG_DIR", str(user_dir / "job_logs"))).resolve()

    cors_allow_origins = _csv_env("CORS_ALLOW_ORIGINS")
    cors_allow_origin_regex = None
    if run_mode == "local":
        cors_allow_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    elif not cors_allow_origins:
        raise RuntimeError("server 模式必须通过 CORS_ALLOW_ORIGINS 配置允许的前端来源白名单。")

    auth_enabled = _bool_env("TOXHERB_AUTH_ENABLED", run_mode == "server")
    api_token = os.getenv("API_TOKEN")
    if auth_enabled and not api_token:
        raise RuntimeError("启用鉴权时必须配置 API_TOKEN。")

    for path in (user_dir, output_dir, job_db.parent, job_log_dir):
        path.mkdir(parents=True, exist_ok=True)

    return Settings(
        root_dir=root_dir,
        run_mode=run_mode,
        cors_allow_origins=cors_allow_origins,
        cors_allow_origin_regex=cors_allow_origin_regex,
        max_concurrent_jobs=_int_env("MAX_CONCURRENT_JOBS", 1, minimum=1),
        script_timeout_seconds=_int_env("TOXHERB_SCRIPT_TIMEOUT_SECONDS", 3600, minimum=30),
        output_dir=output_dir,
        user_dir=user_dir,
        job_db=job_db,
        job_log_dir=job_log_dir,
        enable_visualization=_bool_env("TOXHERB_ENABLE_VISUALIZATION", True),
        auth_enabled=auth_enabled,
        api_token=api_token,
        rate_limit_per_minute=_int_env("TOXHERB_RATE_LIMIT_PER_MINUTE", 60, minimum=1),
    )
