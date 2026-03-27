from pydantic import BaseModel, Field, model_validator
from typing import List


class TrainData(BaseModel):
    """Payload de entrada para o endpoint de treino."""
    timestamps: List[int] = Field(..., description="Unix timestamps")
    values: List[float] = Field(..., description="Measured values")

    @model_validator(mode="after")
    def validate_lengths(self) -> "TrainData":
        """Valida consistência estrutural entre timestamps e values."""
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have the same length.")
        return self


class TrainResponse(BaseModel):
    """Resposta do endpoint de treino com identificação da versão gerada."""
    series_id: str
    version: str
    points_used: int
