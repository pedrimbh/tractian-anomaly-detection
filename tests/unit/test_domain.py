import pytest
from app.domain.model import AnomalyDetectionModel


# ──────────────────────────────────────────
# AnomalyDetectionModel
# ──────────────────────────────────────────

def test_model_detects_anomaly():
    model = AnomalyDetectionModel()
    model.fit([10.0 + i * 0.1 for i in range(100)])
    assert model.predict(9999.0) is True


def test_model_accepts_normal_value():
    model = AnomalyDetectionModel()
    model.fit([10.0 + i * 0.01 for i in range(100)])
    assert model.predict(10.5) is False


def test_model_constant_std_never_anomaly():
    """Com std=0 o modelo não deve explodir nem marcar anomalia."""
    model = AnomalyDetectionModel()
    model.fit([5.0] * 100)
    assert model.predict(9999.0) is False


def test_model_fit_sets_mean_and_std():
    model = AnomalyDetectionModel()
    model.fit([10.0, 20.0])
    assert model.mean == 15.0
    assert model.std == 5.0
