from fastapi import FastAPI
from app.core.logging import setup_logging
from app.middleware.audit_log import AuditLogMiddleware
from app.api.v1.router import router as v1_router

setup_logging()

app: FastAPI = FastAPI(
    title="Time Series Anomaly Detection API",
    version="0.0.0",
    description="Anomaly detection on univariate time series data.",
)

# Middlewares
app.add_middleware(AuditLogMiddleware)

# Routers
app.include_router(v1_router)
