from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque

from fastapi import HTTPException, Request

from .config import Settings


@dataclass
class RateLimiter:
    requests_per_minute: int
    _events: dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: Lock = field(default_factory=Lock)

    def check(self, key: str) -> None:
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            events = self._events[key]
            while events and events[0] < window_start:
                events.popleft()
            if len(events) >= self.requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "code": "rate_limit_exceeded",
                        "message": "请求过于频繁，请稍后再试。",
                        "suggestion": "降低提交频率，或在 server 模式下调整 TOXHERB_RATE_LIMIT_PER_MINUTE。",
                    },
                )
            events.append(now)


class SecurityGuard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    def require_api_access(self, request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        if self.settings.run_mode == "server":
            self.rate_limiter.check(client_host)

        if not self.settings.auth_enabled:
            return

        expected = self.settings.api_token
        header_token = request.headers.get("x-api-token")
        auth_header = request.headers.get("authorization", "")
        bearer_token = ""
        if auth_header.lower().startswith("bearer "):
            bearer_token = auth_header.split(" ", 1)[1].strip()

        if not expected or header_token == expected or bearer_token == expected:
            return

        raise HTTPException(
            status_code=401,
            detail={
                "code": "unauthorized",
                "message": "缺少或无效的 API Token。",
                "suggestion": "请在请求头中提供 X-API-Token 或 Authorization: Bearer <token>。",
            },
        )
