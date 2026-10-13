from typing import Any

from fastapi import APIRouter, Depends

from app.predictors import Predictor, get_predictor
from app.registry import Registry, get_registry
from app.schemas import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(
    registry: Registry = Depends(get_registry), predictor: Predictor = Depends(get_predictor)
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=predictor.info,
        features=len(registry.features),
        targets=list(registry.targets),
    )


@router.get("/schema")
def feature_schema(registry: Registry = Depends(get_registry)) -> dict[str, Any]:
    """Input feature definitions used to auto-generate the patient form."""
    return registry.raw_schema


@router.get("/targets")
def targets(registry: Registry = Depends(get_registry)) -> dict[str, Any]:
    """Prediction targets and their mapping to 3D anatomy mesh names."""
    return registry.raw_vessels
