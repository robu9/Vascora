"""SHAP explanations of the full ensemble in log-odds space over the schema features."""

from typing import Any

import numpy as np
import pandas as pd
import shap

from vascora_ml import data
from vascora_ml.ensemble import VascoraModel

_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class ShapExplainer:
    """Permutation SHAP with an interventional background of real (dev-set) patients.

    Explains the end-to-end model (feature engineering, chaining, voting and calibration) for
    all targets in one multi-output pass, so chained targets add no extra cost. Attributions
    are additive in log-odds: base_value + sum(values) == logit(probability).
    """

    def __init__(
        self,
        model: VascoraModel,
        background: pd.DataFrame,
        n_background: int = 30,
        permutations: int = 2,
        seed: int = 0,
    ):
        self.model = model
        self.keys = list(model.feature_keys_)
        self.targets = list(model.targets)
        bg = background[self.keys]
        if len(bg) > n_background:
            bg = bg.sample(n_background, random_state=seed)
        self.background = bg.reset_index(drop=True)
        self.permutations = permutations
        self.seed = seed
        self.base_values = dict(
            zip(self.targets, self._logits(self.background.to_numpy()).mean(axis=0), strict=True)
        )

    def _logits(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict_proba(pd.DataFrame(X, columns=self.keys))
        return np.column_stack([logit(probs[t]) for t in self.targets])

    def explain(self, Xb: pd.DataFrame, permutations: int | None = None) -> dict[str, Any]:
        """Return {target: shap.Explanation} for every row of `Xb`."""
        perms = permutations or self.permutations
        masker = shap.maskers.Independent(self.background.to_numpy(),
                                          max_samples=len(self.background))
        explainer = shap.PermutationExplainer(self._logits, masker, feature_names=self.keys,
                                              seed=self.seed)
        ev = explainer(Xb[self.keys].to_numpy(), max_evals=2 * len(self.keys) * perms + 1,
                       silent=True)
        out = {}
        for j, t in enumerate(self.targets):
            e = ev[:, :, j]
            e.feature_names = self.keys
            out[t] = e
        return out


def explanation_records(ev, Xb: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-row additive contributions keyed by schema feature, with decoded raw values."""
    specs = {f.key: f for f in data.features()}
    base = np.broadcast_to(np.asarray(ev.base_values, dtype=float), (len(Xb),))
    rows = []
    for i in range(len(Xb)):
        contribs = [
            {
                "key": k,
                "label": specs[k].label,
                "category": specs[k].category,
                "value": data.decode_value(k, Xb.iloc[i][k]),
                "contribution": float(ev.values[i, j]),
            }
            for j, k in enumerate(ev.feature_names)
        ]
        contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        rows.append({"base_value": float(base[i]), "contributions": contribs})
    return rows


def global_importance(ev) -> pd.DataFrame:
    """Mean |SHAP| per feature plus category, sorted descending."""
    specs = {f.key: f for f in data.features()}
    df = pd.DataFrame(
        {
            "key": ev.feature_names,
            "label": [specs[k].label for k in ev.feature_names],
            "category": [specs[k].category for k in ev.feature_names],
            "mean_abs_shap": np.abs(ev.values).mean(axis=0),
            "mean_shap": ev.values.mean(axis=0),
        }
    )
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
