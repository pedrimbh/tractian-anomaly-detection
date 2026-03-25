from pydantic import BaseModel, Field, model_validator
from typing import List
from app.core.config import settings


class TrainData(BaseModel):
    """Payload de entrada para o endpoint de treino."""
    timestamps: List[int] = Field(..., description="Unix timestamps")
    values: List[float] = Field(..., description="Measured values")

    @model_validator(mode="after")
    def validate_data(self) -> "TrainData":
        """Valida consistência entre timestamps e values, mínimo de pontos e variância."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have the same length.")
        if len(self.values) < settings.min_training_points:
            raise ValueError(
                f"At least {settings.min_training_points} data points are required for training."
            )
        if len(set(self.values)) == 1:
            raise ValueError(
                "Training data is constant — model cannot learn a meaningful distribution."
            )
        return self


class TrainResponse(BaseModel):
    """Resposta do endpoint de treino com identificação da versão gerada."""
    series_id: str
    version: str
    points_used: int
