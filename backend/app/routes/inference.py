from fastapi import APIRouter, Depends, Query
from vascora_ml.counterfactual import suggest

from app.predictors import Predictor, get_predictor
from app.registry import Registry, get_registry
from app.schemas import (
    AnalyzeResponse,
    CounterfactualResponse,
    ExplainResponse,
    PredictResponse,
)
from app.services import build_explanations, build_prediction, resolve_targets

router = APIRouter(tags=["inference"])

PatientFeatures = get_registry().patient_model

TargetsQuery = Query(None, description="Subset of targets to explain; defaults to all.")


@router.post("/predict", response_model=PredictResponse)
def predict(
    patient: PatientFeatures,  # type: ignore[valid-type]
    registry: Registry = Depends(get_registry),
    predictor: Predictor = Depends(get_predictor),
) -> PredictResponse:
    values, imputed = registry.complete(patient.model_dump())
    return build_prediction(registry, predictor, values, imputed)


@router.post("/explain", response_model=ExplainResponse)
def explain(
    patient: PatientFeatures,  # type: ignore[valid-type]
    targets: list[str] | None = TargetsQuery,
    registry: Registry = Depends(get_registry),
    predictor: Predictor = Depends(get_predictor),
) -> ExplainResponse:
    values, _ = registry.complete(patient.model_dump())
    return ExplainResponse(
        model=predictor.info,
        explanations=build_explanations(registry, predictor, values, targets),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    patient: PatientFeatures,  # type: ignore[valid-type]
    targets: list[str] | None = TargetsQuery,
    registry: Registry = Depends(get_registry),
    predictor: Predictor = Depends(get_predictor),
) -> AnalyzeResponse:
    """Prediction and explanations in one round trip for live what-if updates."""
    values, imputed = registry.complete(patient.model_dump())
    return AnalyzeResponse(
        prediction=build_prediction(registry, predictor, values, imputed),
        explanations=build_explanations(registry, predictor, values, targets),
    )


@router.post("/counterfactual", response_model=CounterfactualResponse)
def counterfactual(
    patient: PatientFeatures,  # type: ignore[valid-type]
    target: str = Query("CAD", description="Target the hints try to lower."),
    registry: Registry = Depends(get_registry),
    predictor: Predictor = Depends(get_predictor),
) -> CounterfactualResponse:
    """Changes to modifiable features that the model scores as lower risk."""
    chosen = resolve_targets(registry, [target])[0]
    values, _ = registry.complete(patient.model_dump())
    result = suggest(values, lambda rows: predictor.predict_many(rows, chosen))
    return CounterfactualResponse(model=predictor.info, target=chosen, **result)
