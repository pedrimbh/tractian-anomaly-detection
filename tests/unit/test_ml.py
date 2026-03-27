import pytest
from app.ml.base import ModelTrainingError
from app.ml.three_sigma import ThreeSigmaDetector


# ──────────────────────────────────────────
# ThreeSigmaDetector
# ──────────────────────────────────────────

@pytest.mark.unit
def test_model_detects_anomaly():
    model = ThreeSigmaDetector()
    model.fit([10.0 + i * 0.1 for i in range(100)])
    assert model.predict(9999.0) is True


@pytest.mark.unit
def test_model_accepts_normal_value():
    model = ThreeSigmaDetector()
    model.fit([10.0 + i * 0.01 for i in range(100)])
    assert model.predict(10.5) is False


@pytest.mark.unit
def test_model_rejects_constant_data():
    """Dados constantes devem levantar ModelTrainingError."""
    model = ThreeSigmaDetector()
    with pytest.raises(ModelTrainingError):
        model.fit([5.0] * 100)


@pytest.mark.unit
def test_model_rejects_insufficient_data():
    """Menos de min_training_points deve levantar ModelTrainingError."""
    model = ThreeSigmaDetector()
    with pytest.raises(ModelTrainingError):
        model.fit([10.0, 20.0])


@pytest.mark.unit
def test_model_fit_sets_mean_and_std():
    model = ThreeSigmaDetector()
    model.fit([10.0] * 5 + [20.0] * 5)  # mean=15.0, std=5.0
    assert model.mean == 15.0
    assert model.std == 5.0
