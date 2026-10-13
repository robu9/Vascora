"""Trained ensemble from the ML pipeline (`ml/artifacts/vascora_model.joblib`)."""

import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from vascora_ml.data import encode_base

from app.registry import Registry
from app.schemas import FeatureContribution, ModelInfo, TargetExplanation

_CACHE_SIZE = 64


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class TrainedPredictor:
    def __init__(self, registry: Registry, path: Path):
        bundle = joblib.load(path)
        self.registry = registry
        self.model = bundle["model"]
        self.explainer = bundle["explainer"]
        missing = set(registry.targets) - set(self.model.targets)
        if missing:
            raise ValueError(f"Model artifact lacks targets: {sorted(missing)}")
        if list(self.model.feature_keys_) != list(registry.features):
            raise ValueError("Model artifact was trained on a different feature schema")
        self.info = ModelInfo(
            name="calibrated-ensemble", version=str(bundle["version"]), is_placeholder=False
        )
        self.thresholds = {t: float(self.model.thresholds[t]) for t in registry.targets}
        self._shap_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()

    def _encode(self, values: dict[str, Any]) -> pd.DataFrame:
        return encode_base(pd.DataFrame([{k: values[k] for k in self.registry.features}]))

    def predict(
        self, values: dict[str, Any]
    ) -> dict[str, tuple[float, tuple[float, float] | None]]:
        row = self._encode(values).iloc[0].to_dict()
        return {t: (p, (lo, hi)) for t, (p, lo, hi) in
                self.model.predict_row_interval(row).items()}

    def predict_many(self, rows: list[dict[str, Any]], target: str):
        frame = pd.DataFrame(rows)[list(self.registry.features)]
        return self.model.predict_proba(encode_base(frame))[target]

    def _shap(self, values: dict[str, Any]) -> dict[str, Any]:
        key = tuple(values[k] for k in self.registry.features)
        if key in self._shap_cache:
            self._shap_cache.move_to_end(key)
            return self._shap_cache[key]
        result = self.explainer.explain(self._encode(values))
        self._shap_cache[key] = result
        if len(self._shap_cache) > _CACHE_SIZE:
            self._shap_cache.popitem(last=False)
        return result

    def explain(self, values: dict[str, Any], targets: list[str]) -> dict[str, TargetExplanation]:
        shap_values = self._shap(values)
        explanations = {}
        for target in targets:
            ev = shap_values[target]
            base = float(ev.base_values[0])
            totals: dict[str, float] = defaultdict(float)
            items = []
            for j, key in enumerate(ev.feature_names):
                spec = self.registry.features[key]
                c = float(ev.values[0, j])
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
            output = base + sum(fc.contribution for fc in items)
            explanations[target] = TargetExplanation(
                target=target,
                method="permutation SHAP (log-odds, end-to-end ensemble)",
                base_value=base,
                output_value=output,
                probability=_sigmoid(output),
                contributions=items,
                category_totals=dict(totals),
            )
        return explanations
