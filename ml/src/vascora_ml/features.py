"""Feature engineering applied inside every model pipeline (so it is refit per CV fold)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_EPS = 1e-6

RISK_FACTORS = ["dm", "htn", "current_smoker", "fh", "dlp", "obesity"]
ECG_FINDINGS = [
    "q_wave", "st_elevation", "st_depression", "t_inversion", "lvh", "poor_r_progression"
]
RATIOS = {"tg_hdl": ("tg", "hdl"), "ldl_hdl": ("ldl", "hdl"), "nlr": ("neut", "lymph"),
          "bun_cr": ("bun", "cr")}


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Base-encoded features -> model matrix.

    Adds clinically motivated ratios/counts, one-hot encodes the nominal BBB code and drops
    columns that are constant in the training fold. Extra columns (e.g. chained parent-target
    probabilities) pass through unchanged. Works on numpy internally because it runs on
    every prediction and SHAP evaluation.
    """

    def __init__(self, engineered: bool = True):
        self.engineered = engineered

    def _expand(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        cols = list(X.columns)
        A = X.to_numpy(dtype=float)
        ix = {c: i for i, c in enumerate(cols)}
        parts, names = [], []
        if "bbb" in ix:
            code = np.round(A[:, ix["bbb"]])
            parts += [(code == 1).astype(float), (code == 2).astype(float)]
            names += ["bbb_lbbb", "bbb_rbbb"]
        if self.engineered:
            for name, (num, den) in RATIOS.items():
                parts.append(A[:, ix[num]] / (A[:, ix[den]] + _EPS))
                names.append(name)
            parts.append(A[:, [ix[c] for c in RISK_FACTORS]].sum(axis=1))
            parts.append(A[:, [ix[c] for c in ECG_FINDINGS]].sum(axis=1))
            female = np.round(A[:, ix["sex"]]) == 1
            age = A[:, ix["age"]]
            parts.append(np.where(female, age >= 55, age >= 45).astype(float))
            names += ["risk_factor_count", "ecg_finding_count", "age_risk"]
        keep = [i for i, c in enumerate(cols) if c != "bbb"]
        out = np.column_stack([A[:, keep], *parts]) if parts else A[:, keep]
        return out, [cols[i] for i in keep] + names

    def fit(self, X: pd.DataFrame, y=None):
        A, names = self._expand(pd.DataFrame(X))
        self.columns_in_ = list(pd.DataFrame(X).columns)
        self.kept_idx_ = [i for i in range(A.shape[1]) if np.unique(A[:, i]).size > 1]
        self.kept_ = [names[i] for i in self.kept_idx_]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        if list(X.columns) != self.columns_in_:
            X = X[self.columns_in_]
        A, _ = self._expand(X)
        return A[:, self.kept_idx_]

    def transform_row(self, row: dict) -> np.ndarray:
        """Single-patient path used by the API: no DataFrame construction."""
        A = np.array([[float(row[c]) for c in self.columns_in_]])
        ix = {c: i for i, c in enumerate(self.columns_in_)}
        parts = []
        if "bbb" in ix:
            code = round(A[0, ix["bbb"]])
            parts += [code == 1, code == 2]
        if self.engineered:
            for num, den in RATIOS.values():
                parts.append(A[0, ix[num]] / (A[0, ix[den]] + _EPS))
            parts.append(sum(A[0, ix[c]] for c in RISK_FACTORS))
            parts.append(sum(A[0, ix[c]] for c in ECG_FINDINGS))
            female = round(A[0, ix["sex"]]) == 1
            parts.append(A[0, ix["age"]] >= (55 if female else 45))
        keep = [i for i, c in enumerate(self.columns_in_) if c != "bbb"]
        out = np.array([*(A[0, i] for i in keep), *(float(p) for p in parts)])
        return out[self.kept_idx_][None, :]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.kept_, dtype=object)
