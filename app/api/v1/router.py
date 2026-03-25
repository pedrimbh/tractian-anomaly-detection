from fastapi import APIRouter
from app.routers import healthcheck, prediction, training

router = APIRouter()
router.include_router(healthcheck.router)
router.include_router(prediction.router)
router.include_router(training.router)
