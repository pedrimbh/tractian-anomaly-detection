# Time Series Anomaly Detection API

REST API for anomaly detection on univariate time series data.
Supports multiple independent series, model versioning, real-time predictions, and audit logging.

---

## Quick Start

### Local (Python)

```bash
# 1. Clone the repo
git clone https://github.com/pedrimbh/tractian-anomaly-detection
cd tractian-anomaly-detection

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies (runtime + dev)
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env

# 5. Run the API
uvicorn app.main:app --reload
```

API available at: **http://localhost:8000**
Interactive docs: **http://localhost:8000/docs**

### Docker

```bash
docker build --target production -t tractian-api .
docker run -p 8000:8000 tractian-api
```

---

## Endpoints

### `POST /fit/{series_id}` — Train a model

```bash
curl -X POST http://localhost:8000/fit/sensor_xyz \
  -H "Content-Type: application/json" \
  -d '{
    "timestamps": [1700000000, 1700000060, 1700000120, 1700000180, 1700000240, 1700000300, 1700000360, 1700000420, 1700000480, 1700000540],
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

### `GET /plot/{series_id}` — Visualize trained model

Opens an interactive Plotly chart in the browser showing:
- Histogram of training data points
- Fitted Gaussian curve N(μ, σ)
- Mean line and anomaly threshold (μ + 3σ)
- Shaded anomaly region

```bash
# Latest version (train first!)
open http://localhost:8000/plot/sensor_xyz

# Specific version
open http://localhost:8000/plot/sensor_xyz?version=v2
```

Returns a standalone HTML page (Plotly JS from CDN). Works directly in any browser — supports zoom, hover with exact values, and pan.

> **Note:** The model must be trained before accessing the plot. Call `POST /fit/{series_id}` first, then open the plot URL.

---

### `GET /healthcheck` — System metrics

```bash
curl http://localhost:8000/healthcheck
```

```json
{
  "series_trained": 3,
  "total_models": 7,
  "inference_latency_ms": { "avg": 0.85, "p50": 0.78, "p95": 1.42, "p99": 1.91 },
  "training_latency_ms": { "avg": 2.10, "p50": 1.95, "p95": 3.87, "p99": 4.20 }
}
```

---

## Input Validation

The API rejects training requests with:
- Fewer than **10 data points** (configurable via `MIN_TRAINING_POINTS`)
- **Constant** series (zero variance)
- Mismatched `timestamps` and `values` lengths

---

## Audit Logging

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

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest -m unit -v

# Integration tests only
pytest -m integration -v
```

---

## Load Testing

Load tests simulate real production traffic using [Locust](https://locust.io/) — multiple concurrent users sending independent HTTP requests with full middleware, cache, and Gunicorn overhead.

### Running

Use Docker to start the API — Gunicorn manages 4 worker processes, which is closer to a real production environment and exercises cross-process locking and concurrency properly.

**Terminal 1 — start the API with Docker (Gunicorn + 4 workers):**
```bash
docker build --target production -t tractian-api .
docker run --rm -p 8000:8000 tractian-api
```

**Terminal 2 — start Locust:**
```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

Open **http://localhost:8089** in the browser and configure:

| Field | Description | Example |
|---|---|---|
| Number of users | Total concurrent virtual users | `50` |
| Spawn rate | Users added per second until target | `10` |
| Host | API base URL (no trailing path) | `http://localhost:8000` |

Click **Start** to begin. The UI shows RPS, median/p95/p99 latency, and failure rate per endpoint (`/fit` and `/predict`) in real time.

> **Why Docker?** Gunicorn spawns 4 independent worker processes — each with its own memory and event loop. This exercises cross-process FileLock coordination, cache isolation between workers, and realistic OS scheduling, which a single `uvicorn` process cannot replicate.

### Headless mode (CI / terminal only)

```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000 \
  --users 50 --spawn-rate 10 --run-time 60s --headless
```

### What the test does

Each virtual user picks a random sensor (`sensor_1` to `sensor_10`), trains a model on startup, then alternates equally between:
- `POST /predict/{series_id}` — prediction with a random value
- `POST /fit/{series_id}` — model retraining

This exercises cache hits and misses, concurrent writes, and model versioning under load.

---

## Project Structure

```
tractian-anomaly-detection/
├── app/
│   ├── main.py                   # App factory, middleware and router registration
│   ├── api/
│   │   └── v1/
│   │       └── router.py         # Assembles all v1 routers (versioning-ready)
│   ├── routers/
│   │   ├── training.py           # POST /fit/{series_id}
│   │   ├── prediction.py         # POST /predict/{series_id}
│   │   ├── healthcheck.py        # GET /healthcheck
│   │   └── plot.py               # GET /plot/{series_id}
│   ├── services/
│   │   ├── training_service.py   # Orchestrates fit: versioning, storage, metrics, logging
│   │   └── prediction_service.py # Orchestrates predict: cache, inference, metrics, logging
│   ├── middleware/
│   │   └── audit_log.py          # Structured audit logging for all requests
│   ├── core/
│   │   ├── config.py             # Centralized settings via pydantic-settings
│   │   ├── logging.py            # structlog setup with file rotation
│   │   ├── metrics.py            # Thread-safe in-memory latency collector
│   │   ├── model_cache.py        # Thread-safe in-memory model cache
│   │   └── series_lock.py        # FileLock registry (cross-process per-series locking)
│   ├── ml/
│   │   ├── base.py               # AnomalyDetector (ABC) + ModelTrainingError
│   │   └── three_sigma.py        # ThreeSigmaDetector — 3-sigma rule implementation
│   ├── infra/
│   │   └── storage.py            # Disk I/O: save/load models + versioning
│   └── schemas/
│       ├── training.py           # TrainData, TrainResponse
│       ├── prediction.py         # PredictData, PredictResponse
│       └── health.py             # HealthCheckResponse, Metrics
├── tests/
│   ├── unit/
│   │   └── test_ml.py            # ML model unit tests (no I/O)
│   ├── integration/
│   │   └── test_api.py           # Full endpoint tests via TestClient
│   └── load/
│       └── locustfile.py         # Locust load test — concurrent users via HTTP
├── logs/                         # Auto-created, gitignored
├── models_store/                 # Auto-created, gitignored
├── .env.example
├── .gitignore
├── Dockerfile
└── pyproject.toml
```

> **Architecture note:** The `api/v1/` layer exists as a versioning-ready structure.
> Routes are defined in `routers/` and assembled in `api/v1/router.py`, so introducing
> a `v2` is as simple as adding `api/v2/router.py` and including new or reused routers.

---

## Model Details

The anomaly detector uses the **3-sigma rule**:
- Learns **mean (μ)** and **standard deviation (σ)** from training data
- A point `x` is anomalous if: `x > μ + 3σ`
- Each `series_id` maintains independent models with auto-incremented versions (`v1`, `v2`, ...)
- Models persist to disk as JSON and survive restarts
