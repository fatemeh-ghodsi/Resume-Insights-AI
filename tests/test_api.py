from fastapi.testclient import TestClient
import joblib

from resume_insights.api import create_app
from resume_insights.model import build_classifier


def _write_model(path):
    texts = [
        "Python FastAPI PostgreSQL backend API developer",
        "Python Django SQL backend software engineer",
        "REST API Docker Python server developer",
        "Registered nurse patient care hospital clinical",
        "Nursing healthcare patient treatment clinical care",
        "Hospital nurse medical patient support",
    ]
    labels = ["Python Developer"] * 3 + ["Healthcare"] * 3
    pipeline = build_classifier()
    pipeline.fit(texts, labels)
    joblib.dump(
        {"pipeline": pipeline, "metadata": {"model": "TF-IDF + LinearSVC"}},
        path,
    )


def test_health_endpoint_reports_model_not_loaded(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.joblib"))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_endpoint_loads_model_lazily(tmp_path):
    model_path = tmp_path / "model.joblib"
    _write_model(model_path)
    client = TestClient(create_app(model_path))

    response = client.post(
        "/predict",
        json={"text": "Experienced Python developer building FastAPI and PostgreSQL services"},
    )

    assert response.status_code == 200
    assert response.json()["predicted_category"] == "Python Developer"
    assert response.json()["model"] == "TF-IDF + LinearSVC"
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_returns_service_unavailable_without_model(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.joblib"))
    response = client.post(
        "/predict",
        json={"text": "Experienced Python developer building FastAPI and PostgreSQL services"},
    )
    assert response.status_code == 503


def test_predict_validates_short_text(tmp_path):
    model_path = tmp_path / "model.joblib"
    _write_model(model_path)
    client = TestClient(create_app(model_path))
    response = client.post("/predict", json={"text": "too short"})
    assert response.status_code == 422
