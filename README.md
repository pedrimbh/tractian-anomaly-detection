# 🧠 Time Series Anomaly Detection API

REST API for anomaly detection on univariate time series data.  
Supports multiple independent series, model versioning, real-time predictions, audit logging, and data drift detection.

---

## 🚀 Quick Start

### Local (Python)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/tractian-anomaly-detection
cd tractian-anomaly-detection

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env

# 5. Run the API
uvicorn app.main:app --reload
```

API available at: **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

### Docker

```bash
docker build -t anomaly-api .
docker run -p 8000:8000 \
  -v $(pwd)/models_store:/app/models_store \
  -v $(pwd)/logs:/app/logs \
  anomaly-api
```

---

## 📡 Endpoints

### `POST /fit/{series_id}` — Train a model

```bash
curl -X POST http://localhost:8000/fit/sensor_xyz \
  -H "Content-Type: application/json" \
  -d '{
    "timestamps": [1700000000, 1700000060, 1700000120],
    "values": [10.5, 11.2, 10.8, 10.3, 11.0, 10.7, 10.4, 11.1, 10.6, 10.9]
  }'
```

```json
{ "series_id": "sensor_xyz", "version": "v1", "points_used": 10 }
```

---

### `POST /predict/{series_id}` — Detect anomaly

```bash
curl -X POST http://localhost:8000/predict/sensor_xyz \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "1700001000", "value": 99.9}'
```

```json
{ "anomaly": true, "model_version": "v1" }
```

Use `?version=v1` to predict against a specific model version.

---

### `GET /healthcheck` — System metrics

```bash
curl http://localhost:8000/healthcheck
```

```json
{
  "series_trained": 3,
  "inference_latency_ms": { "avg": 0.85, "p95": 1.42 },
  "training_latency_ms": { "avg": 2.10, "p95": 3.87 }
}
```

---

### `GET /drift/{series_id}` — Data drift analysis

Compares the distribution of recent prediction inputs against the trained distribution.  
Useful to detect when a sensor starts measuring differently and the model should be retrained.

```bash
curl http://localhost:8000/drift/sensor_xyz?recent_n=100
```

```json
{
  "series_id": "sensor_xyz",
  "model_version": "v1",
  "trained_mean": 10.42,
  "trained_std": 0.18,
  "recent_mean": 14.87,
  "recent_std": 0.21,
  "mean_shift": 4.45,
  "drift_detected": true,
  "recommendation": "consider retraining — input distribution has shifted significantly"
}
```

---

## ✅ Input Validation

The API rejects training requests with:
- Fewer than **10 data points** (configurable via `MIN_TRAINING_POINTS`)
- **Constant** series (zero variance)
- Mismatched `timestamps` and `values` lengths

---

## 📋 Audit Logging

Every request is automatically logged with:

| Field | Description |
|---|---|
| `timestamp` | When the request was processed |
| `request_id` | Unique UUID per request |
| `request_payload` | Exact JSON body received |
| `response_payload` | Exact JSON body returned |
| `status_code` | HTTP response code |
| `latency_ms` | Total request duration |

Logs are written to both **stdout** (for Docker/cloud collectors) and **`logs/app.log`** (with daily rotation, 7 days retention).

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (no FastAPI needed)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v
```

---

## 📊 Load Benchmark

```bash
# Start the API first
uvicorn app.main:app &

# Run 100 parallel inferences
python scripts/benchmark.py
```

Example output:
```
──────────────────────────────────
  Requisições  : 100
  Tempo total  : 298.4 ms
  Throughput   : 335.1 req/s
  Latência avg : 54.2 ms
  Latência p50 : 51.8 ms
  Latência p95 : 87.3 ms
  Latência p99 : 101.2 ms
  Latência max : 108.4 ms
──────────────────────────────────
```

---

## 🗂️ Project Structure

```
tractian-anomaly-detection/
├── app/
│   ├── main.py                        # App factory + middleware + router registration
│   ├── api/
│   │   ├── routers/
│   │   │   ├── training.py            # POST /fit/{series_id}
│   │   │   ├── prediction.py          # POST /predict/{series_id}
│   │   │   ├── healthcheck.py         # GET /healthcheck
│   │   │   └── drift.py              # GET /drift/{series_id}
│   │   ├── middleware/
│   │   │   └── audit_log.py          # Intercepts all requests for structured logging
│   │   └── schemas.py                # Pydantic request/response models
│   ├── core/
│   │   ├── config.py                 # Centralized settings via pydantic-settings
│   │   ├── logging.py                # structlog setup with file rotation
│   │   └── metrics.py                # Thread-safe in-memory latency collector
│   ├── domain/
│   │   ├── model.py                  # AnomalyDetectionModel (3-sigma, pure Python)
│   │   └── drift.py                  # Drift detection logic (pure Python)
│   └── infra/
│       └── storage.py                # All disk I/O: save/load models + prediction history
├── tests/
│   ├── unit/
│   │   └── test_domain.py            # Tests domain logic with no FastAPI or I/O
│   └── integration/
│       └── test_api.py               # Tests all endpoints with TestClient
├── scripts/
│   └── benchmark.py                  # 100 parallel requests load test
├── logs/                             # Auto-created, gitignored
├── models_store/                     # Auto-created, gitignored
├── .env.example
├── .gitignore
├── Dockerfile
└── requirements.txt
```

---

## 🔧 Model Details

The anomaly detector uses the **3-sigma rule**:
- Learns **mean (μ)** and **standard deviation (σ)** from training data
- A point `x` is anomalous if: `x > μ + 3σ`
- Each `series_id` maintains independent models with auto-incremented versions (`v1`, `v2`, ...)
- Models persist to disk as JSON and survive restarts
