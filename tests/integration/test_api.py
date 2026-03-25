from typing import Any

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SERIES_ID = "integration_sensor_01"

TRAIN_PAYLOAD: dict[str, Any] = {
    "timestamps": list(range(1700000000, 1700000100)),
    "values": [10.0 + (i % 5) * 0.1 for i in range(100)],
}


# ──────────────────────────────────────────
# Training
# ──────────────────────────────────────────

def test_fit_success():
    r = client.post(f"/fit/{SERIES_ID}", json=TRAIN_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["series_id"] == SERIES_ID
    assert data["points_used"] == 100
    assert data["version"].startswith("v")


def test_fit_versioning():
    r1 = client.post(f"/fit/{SERIES_ID}", json=TRAIN_PAYLOAD)
    r2 = client.post(f"/fit/{SERIES_ID}", json=TRAIN_PAYLOAD)
    v1 = int(r1.json()["version"][1:])
    v2 = int(r2.json()["version"][1:])
    assert v2 == v1 + 1


def test_fit_rejects_insufficient_data():
    payload: dict[str, Any] = {"timestamps": [1, 2], "values": [1.0, 2.0]}
    assert client.post(f"/fit/{SERIES_ID}", json=payload).status_code == 422


def test_fit_rejects_constant_data():
    payload: dict[str, Any] = {"timestamps": list(range(100)), "values": [5.0] * 100}
    assert client.post(f"/fit/{SERIES_ID}", json=payload).status_code == 422


def test_fit_rejects_mismatched_lengths():
    payload: dict[str, Any] = {"timestamps": [1, 2, 3], "values": [1.0, 2.0]}
    assert client.post(f"/fit/{SERIES_ID}", json=payload).status_code == 422


# ──────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────

def test_predict_normal_value():
    client.post(f"/fit/{SERIES_ID}", json=TRAIN_PAYLOAD)
    r = client.post(f"/predict/{SERIES_ID}", json={"timestamp": "1700001000", "value": 10.2})
    assert r.status_code == 200
    assert r.json()["anomaly"] is False


def test_predict_anomalous_value():
    client.post(f"/fit/{SERIES_ID}", json=TRAIN_PAYLOAD)
    r = client.post(f"/predict/{SERIES_ID}", json={"timestamp": "1700001001", "value": 9999.0})
    assert r.status_code == 200
    assert r.json()["anomaly"] is True


def test_predict_specific_version():
    r_train = client.post(f"/fit/{SERIES_ID}", json=TRAIN_PAYLOAD)
    version = r_train.json()["version"]
    r = client.post(f"/predict/{SERIES_ID}?version={version}", json={"timestamp": "1700001002", "value": 10.1})
    assert r.status_code == 200
    assert r.json()["model_version"] == version


def test_predict_unknown_series():
    r = client.post("/predict/series_inexistente_xyz", json={"timestamp": "1700001003", "value": 10.0})
    assert r.status_code == 404


# ──────────────────────────────────────────
# Healthcheck
# ──────────────────────────────────────────

def test_healthcheck_shape():
    r = client.get("/healthcheck")
    assert r.status_code == 200
    data = r.json()
    assert "series_trained" in data
    assert "avg" in data["inference_latency_ms"]
    assert "p50" in data["inference_latency_ms"]
    assert "p95" in data["inference_latency_ms"]
    assert "p99" in data["training_latency_ms"]
