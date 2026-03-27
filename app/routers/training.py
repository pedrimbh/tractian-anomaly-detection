import json

from fastapi import APIRouter, HTTPException

from app.ml.base import ModelTrainingError
from app.schemas.training import TrainData, TrainResponse
from app.services import training_service

router = APIRouter(tags=["Training"])


@router.post("/fit/{series_id}", response_model=TrainResponse)
async def fit(series_id: str, body: TrainData) -> TrainResponse:
    """Treina um novo modelo 3-sigma para a série informada e persiste em disco."""
    try:
        result = await training_service.fit(series_id, body.values)
    except ModelTrainingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable, retry later")
    return TrainResponse(series_id=series_id, **result)
