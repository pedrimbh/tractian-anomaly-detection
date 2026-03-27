import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.prediction import PredictData, PredictResponse
from app.services import prediction_service

router = APIRouter(tags=["Prediction"])


@router.post("/predict/{series_id}", response_model=PredictResponse)
async def predict(
    series_id: str,
    body: PredictData,
    version: Optional[str] = Query(default=None),
) -> PredictResponse:
    """Classifica um valor como anomalia ou normal usando o modelo treinado da série."""
    try:
        result = await prediction_service.predict(series_id, body.value, version)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=503, detail="Service temporarily unavailable, retry later")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for series_id='{series_id}'"
            + (f" version='{version}'" if version else ""),
        )
    return PredictResponse(**result)
