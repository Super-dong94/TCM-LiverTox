from __future__ import annotations

import pytest


pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module  # noqa: E402
from backend.predictor import PredictionError  # noqa: E402


client = TestClient(app_module.app)


def test_health_returns_extended_fields() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "data_manifest" in data
    assert "model_manifest" in data
    assert "calibration_manifest" in data
    assert "settings" in data


def test_database_stats_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module.predictor,
        "database_stats",
        lambda: {"ok": True, "stats": {"formulas": 1, "herbs": 2, "compounds": 3}},
    )
    response = client.get("/api/database/stats")
    assert response.status_code == 200
    assert response.json()["stats"]["compounds"] == 3


def test_empty_prediction_error_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: str) -> dict[str, object]:
        raise PredictionError("请输入至少一个方剂名称。")

    monkeypatch.setattr(app_module.predictor, "predict_formula", fail)
    response = client.post("/api/predict/formula", json={"text": ""})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "prediction_error"
    assert "方剂" in detail["message"]


def test_knowledge_visual_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module.predictor,
        "search_knowledge",
        lambda kind, payload: {
            "ok": True,
            "summary": {"matched_compounds": 1},
            "chart_specs": [
                {
                    "id": "knowledge_entity_count_bar",
                    "title": "知识图谱实体数量",
                    "type": "bar",
                    "data": [{"entity": "关联成分", "count": 1}],
                    "encoding": {"x": "entity", "y": "count"},
                    "source_section": "knowledge_overview",
                }
            ],
        },
    )
    response = client.post("/api/visual/knowledge", json={"type": "target", "text": "CYP3A4"})
    assert response.status_code == 200
    assert response.json()["chart_specs"][0]["id"] == "knowledge_entity_count_bar"
