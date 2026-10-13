from typing import Any

from fastapi import HTTPException

from app.predictors import Predictor
from app.registry import Registry
from app.schemas import PredictResponse, TargetExplanation, TargetPrediction


def build_prediction(
    registry: Registry, predictor: Predictor, values: dict[str, Any], imputed: list[str]
) -> PredictResponse:
    raw = predictor.predict(values)

    def target_prediction(key: str) -> TargetPrediction:
        prob, ci = raw[key]
        threshold = predictor.thresholds[key]
        return TargetPrediction(
            target=key,
            label=registry.targets[key].label,
            probability=prob,
            predicted_positive=prob >= threshold,
            threshold=threshold,
            ci=ci,
        )

    return PredictResponse(
        model=predictor.info,
        cad=target_prediction("CAD"),
        vessels={v: target_prediction(v) for v in registry.vessel_keys},
        imputed=imputed,
    )


def resolve_targets(registry: Registry, targets: list[str] | None) -> list[str]:
    if not targets:
        return list(registry.targets)
    unknown = [t for t in targets if t not in registry.targets]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown targets: {unknown}")
    return targets


def build_explanations(
    registry: Registry, predictor: Predictor, values: dict[str, Any], targets: list[str] | None
) -> dict[str, TargetExplanation]:
    return predictor.explain(values, resolve_targets(registry, targets))
