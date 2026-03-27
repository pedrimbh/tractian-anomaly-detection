from fastapi import APIRouter
from app.routers import healthcheck, plot, prediction, training

router = APIRouter()
router.include_router(healthcheck.router)
router.include_router(prediction.router)
router.include_router(training.router)
router.include_router(plot.router)
