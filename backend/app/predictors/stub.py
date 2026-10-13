"""Heuristic placeholder predictor.

Hand-set log-odds weights loosely based on established CAD risk factors. Serves the
full API contract until the trained ensemble lands; not fitted to data and flagged
via `ModelInfo.is_placeholder`.
"""

import math
from collections import defaultdict
from typing import Any

from app.registry import FeatureSpec, Registry
from app.schemas import FeatureContribution, ModelInfo, TargetExplanation

# key -> (log-odds weight, scale). Numeric features are centered on the schema default
# and divided by scale; categorical/ordinal use option index offset from the default.
_CAD_WEIGHTS: dict[str, tuple[float, float]] = {
    "age": (0.6, 10),
    "sex": (-0.4, 1),
    "typical_chest_pain": (1.6, 1),
    "atypical": (-0.8, 1),
    "nonanginal": (-0.9, 1),
    "dm": (0.7, 1),
    "htn": (0.5, 1),
    "current_smoker": (0.4, 1),
    "fh": (0.3, 1),
    "dlp": (0.3, 1),
    "st_depression": (0.6, 1),
    "t_inversion": (0.8, 1),
    "q_wave": (0.6, 1),
    "st_elevation": (0.4, 1),
    "region_rwma": (0.6, 1),
    "ef_tte": (-0.4, 10),
    "fbs": (0.2, 50),
    "ldl": (0.2, 40),
    "hdl": (-0.2, 10),
    "bp": (0.15, 20),
}

_INTERCEPTS = {"CAD": 0.3, "LAD": -0.1, "LCX": -0.7, "RCA": -0.6}

_VESSEL_SCALE = 0.8
_VESSEL_EXTRA: dict[str, dict[str, float]] = {
    "LAD": {"t_inversion": 0.3, "q_wave": 0.3, "region_rwma": 0.2},
    "LCX": {"lvh": 0.2},
    "RCA": {"st_elevation": 0.2, "bbb": 0.1},
}


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _encode(spec: FeatureSpec, value: Any) -> float:
    if spec.type == "number":
        return float(value)
    if spec.type == "binary":
        return float(value)
    if spec.type == "yn":
        return 1.0 if value == "Y" else 0.0
    assert spec.options is not None
    return float(spec.options.index(value))


class StubPredictor:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.info = ModelInfo(name="heuristic-stub", version="0.1.0", is_placeholder=True)
        self.thresholds = {t: 0.5 for t in registry.targets}
        self.weights = self._build_weights()

    def _build_weights(self) -> dict[str, dict[str, tuple[float, float]]]:
        weights: dict[str, dict[str, tuple[float, float]]] = {"CAD": dict(_CAD_WEIGHTS)}
        for vessel in self.registry.vessel_keys:
            w = {k: (wt * _VESSEL_SCALE, scale) for k, (wt, scale) in _CAD_WEIGHTS.items()}
            for key, extra in _VESSEL_EXTRA.get(vessel, {}).items():
                wt, scale = w.get(key, (0.0, 1))
                w[key] = (wt + extra, scale)
            weights[vessel] = w
        return weights

    def _contributions(self, target: str, values: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, spec in self.registry.features.items():
            weight, scale = self.weights[target].get(key, (0.0, 1))
            delta = _encode(spec, values[key]) - _encode(spec, spec.default)
            out[key] = weight * delta / scale
        return out

    def predict(
        self, values: dict[str, Any]
    ) -> dict[str, tuple[float, tuple[float, float] | None]]:
        result = {}
        for target in self.registry.targets:
            z = _INTERCEPTS.get(target, 0.0) + sum(self._contributions(target, values).values())
            result[target] = (_sigmoid(z), None)
        return result

    def predict_many(self, rows: list[dict[str, Any]], target: str):
        return [self.predict(row)[target][0] for row in rows]

    def explain(self, values: dict[str, Any], targets: list[str]) -> dict[str, TargetExplanation]:
        explanations = {}
        for target in targets:
            base = _INTERCEPTS.get(target, 0.0)
            contribs = self._contributions(target, values)
            totals: dict[str, float] = defaultdict(float)
            items = []
            for key, c in contribs.items():
                spec = self.registry.features[key]
                totals[spec.category] += c
                items.append(
                    FeatureContribution(
                        key=key,
                        label=spec.label,
                        category=spec.category,
                        value=values[key],
                        unit=spec.unit,
                        out_of_range=spec.is_out_of_range(values[key]),
                        contribution=c,
                    )
                )
            items.sort(key=lambda fc: abs(fc.contribution), reverse=True)
            output = base + sum(contribs.values())
            explanations[target] = TargetExplanation(
                target=target,
                method="additive-logit (placeholder for SHAP)",
                base_value=base,
                output_value=output,
                probability=_sigmoid(output),
                contributions=items,
                category_totals=dict(totals),
            )
        return explanations
