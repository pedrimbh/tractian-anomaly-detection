from typing import Optional

import numpy as np
import plotly.graph_objects as go
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.core import model_cache
from app.infra import storage

router = APIRouter(tags=["Visualization"])


@router.get("/plot/{series_id}", response_class=HTMLResponse)
async def plot(
    series_id: str,
    version: Optional[str] = Query(default=None),
) -> HTMLResponse:
    """Retorna página HTML com gráfico interativo Plotly do modelo treinado.

    Mostra o histograma dos pontos de treino, a curva gaussiana ajustada,
    a média e o threshold de anomalia (mean + 3σ).
    """
    model = await model_cache.get(series_id, version)
    if model is None:
        resolved = version or await storage.get_latest_version(series_id)
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for series_id='{series_id}'"
            + (f" version='{resolved}'" if resolved else ""),
        )

    resolved_version = version or await storage.get_latest_version(series_id)
    mean, std = model.mean, model.std
    threshold = mean + 3 * std

    x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    fig = go.Figure()

    if model.training_values:
        fig.add_trace(go.Histogram(
            x=model.training_values,
            histnorm="probability density",
            name=f"training data (n={len(model.training_values)})",
            opacity=0.4,
            marker_color="steelblue",
        ))

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        name="fitted distribution",
        line=dict(color="steelblue", width=2),
    ))

    fig.add_vline(x=mean, line_dash="dash", line_color="steelblue",
                  annotation_text=f"mean={mean:.3f}", annotation_position="top left")
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"threshold={threshold:.3f}", annotation_position="top right")
    fig.add_vrect(x0=threshold, x1=float(x[-1]), fillcolor="red", opacity=0.1,
                  line_width=0, name="anomaly region")

    fig.update_layout(
        title=f"{series_id} — {resolved_version}  |  mean={mean:.4f}  std={std:.4f}  threshold={threshold:.4f}",
        xaxis_title="Value",
        yaxis_title="Density",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return HTMLResponse(content=fig.to_html(full_html=True, include_plotlyjs="cdn"))
