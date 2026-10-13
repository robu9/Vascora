"""Final deployable model: calibrated soft-vote ensembles per target, optional chaining."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from vascora_ml import models
from vascora_ml.config import SEED

Params = models.Params


@dataclass
class TargetSpec:
    members: list[tuple[str, Params]]
    parents: list[str] = field(default_factory=list)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class CalibratedSoftVote(ClassifierMixin, BaseEstimator):
    """Soft vote of tuned pipelines with Platt scaling fitted on out-of-fold votes.

    The same out-of-fold probabilities (`oof_proba_`, after calibration) are reused as
    leakage-free inputs for chained child targets, avoiding a second CV layer.
    """

    def __init__(self, members: list[tuple[str, Params]], cv: int = 5, seed: int = SEED):
        self.members = members
        self.cv = cv
        self.seed = seed

    def _vote(self, estimators, X) -> np.ndarray:
        return np.mean([e.predict_proba(X)[:, 1] for e in estimators], axis=0)

    def _vote_row(self, estimators, row: dict) -> float:
        """Single-patient vote: pipeline steps applied without DataFrame wrapping."""
        votes = []
        for est in estimators:
            x = est.named_steps["features"].transform_row(row)
            if "scale" in est.named_steps:
                x = est.named_steps["scale"].transform(x)
            votes.append(est.named_steps["model"].predict_proba(x)[0, 1])
        return float(np.mean(votes))

    def member_probabilities(self, row: dict) -> np.ndarray:
        return np.array([self._vote_row([e], row) for e in self.estimators_])

    def predict_row(self, row: dict) -> float:
        # Calibrator is fitted on logit(mean of member probabilities), so average first.
        z = _logit(np.array([self.member_probabilities(row).mean()]))[:, None]
        return float(self.calibrator_.predict_proba(z)[0, 1])

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        oof = np.zeros(len(y))
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.seed)
        for tr, te in skf.split(X, y):
            fold = [
                models.build(n, p, self.seed).fit(X.iloc[tr], y[tr]) for n, p in self.members
            ]
            oof[te] = self._vote(fold, X.iloc[te])
        self.estimators_ = [models.build(n, p, self.seed).fit(X, y) for n, p in self.members]
        self.calibrator_ = LogisticRegression(C=1e4).fit(_logit(oof)[:, None], y)
        self.oof_proba_ = self.calibrator_.predict_proba(_logit(oof)[:, None])[:, 1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = _logit(self._vote(self.estimators_, X))[:, None]
        return self.calibrator_.predict_proba(z)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_ensemble(members: list[tuple[str, Params]], seed: int = SEED, cv: int = 5):
    return CalibratedSoftVote(members, cv=cv, seed=seed)


def parent_column(target: str) -> str:
    return f"p_{target}"


class VascoraModel:
    """Predicts every target from base-encoded features (see `data.encode_base`).

    Targets are fitted in order; a target with `parents` receives the parents' calibrated
    probabilities as extra inputs (out-of-fold at fit time, so no label leakage).
    `members` are bootstrap refits used for per-patient uncertainty intervals.
    """

    def __init__(
        self,
        specs: dict[str, TargetSpec],
        thresholds: dict[str, float] | None = None,
        seed: int = SEED,
    ):
        self.specs = specs
        self.targets = list(specs)
        self.thresholds = thresholds or {t: 0.5 for t in specs}
        self.seed = seed
        self.estimators_: dict[str, object] = {}
        self.members_: list[dict[str, object]] = []
        self.feature_keys_: list[str] = []

    def _augment(self, Xb: pd.DataFrame, target: str, probs: dict[str, np.ndarray]):
        parents = self.specs[target].parents
        if not parents:
            return Xb
        Xa = Xb.copy()
        for p in parents:
            Xa[parent_column(p)] = probs[p]
        return Xa

    def _fit_set(self, Xb: pd.DataFrame, Y: pd.DataFrame, seed: int) -> dict[str, object]:
        fitted: dict[str, CalibratedSoftVote] = {}
        oof: dict[str, np.ndarray] = {}
        for t in self.targets:
            Xa = self._augment(Xb, t, oof)
            fitted[t] = build_ensemble(self.specs[t].members, seed).fit(Xa, Y[t].to_numpy())
            oof[t] = fitted[t].oof_proba_
        return fitted

    def fit(self, Xb: pd.DataFrame, Y: pd.DataFrame, n_members: int = 0, n_jobs: int = -1):
        self.feature_keys_ = list(Xb.columns)
        self.estimators_ = self._fit_set(Xb, Y, self.seed)
        rng = np.random.default_rng(self.seed)
        boots = [rng.choice(len(Xb), len(Xb), replace=True) for _ in range(n_members)]
        self.members_ = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_set)(
                Xb.iloc[b].reset_index(drop=True), Y.iloc[b].reset_index(drop=True),
                self.seed + i + 1,
            )
            for i, b in enumerate(boots)
        )
        return self

    def _predict_set(self, fitted: dict[str, object], Xb: pd.DataFrame) -> dict[str, np.ndarray]:
        Xb = Xb[self.feature_keys_]
        probs: dict[str, np.ndarray] = {}
        for t in self.targets:
            probs[t] = fitted[t].predict_proba(self._augment(Xb, t, probs))[:, 1]
        return probs

    def predict_proba(self, Xb: pd.DataFrame) -> dict[str, np.ndarray]:
        return self._predict_set(self.estimators_, Xb)

    def predict_row(self, row: dict) -> dict[str, float]:
        """Single-patient prediction used by the API (avoids per-row DataFrame cost)."""
        return {t: p for t, (p, _lo, _hi) in self.predict_row_interval(row).items()}

    def predict_row_interval(self, row: dict, z: float = 1.645) -> dict[str, tuple]:
        """Point prediction plus an approximate 90 % interval per target.

        The interval combines the disagreement of the ensemble members with the calibration
        uncertainty of the Platt mapping; it is the fast counterpart of the bootstrap
        `predict_interval` used for batch reporting.
        """
        out: dict[str, tuple] = {}
        for t in self.targets:
            r = dict(row)
            for p in self.specs[t].parents:
                r[parent_column(p)] = out[p][0]
            est = self.estimators_[t]
            member_p = est.member_probabilities(r)
            mean = float(member_p.mean())
            se = (float(member_p.std(ddof=1) / np.sqrt(len(member_p)))
                  if len(member_p) > 1 else 0.0)
            coef = float(est.calibrator_.coef_[0, 0])
            intercept = float(est.calibrator_.intercept_[0])
            prob = float(_sigmoid(coef * float(_logit(np.array([mean]))[0]) + intercept))
            # Chain rule through logit(mean probability) and the Platt mapping.
            slope = abs(coef) * prob * (1 - prob) / (mean * (1 - mean))
            half = z * slope * se
            out[t] = (prob, float(np.clip(prob - half, 0, 1)),
                      float(np.clip(prob + half, 0, 1)))
        return out

    def predict_target(self, Xb: pd.DataFrame, target: str) -> np.ndarray:
        """Probability for one target (parents computed as needed)."""
        needed = self._ancestors(target) + [target]
        Xb = Xb[self.feature_keys_]
        probs: dict[str, np.ndarray] = {}
        for t in self.targets:
            if t in needed:
                probs[t] = self.estimators_[t].predict_proba(self._augment(Xb, t, probs))[:, 1]
        return probs[target]

    def _ancestors(self, target: str) -> list[str]:
        out: list[str] = []
        for p in self.specs[target].parents:
            out += self._ancestors(p) + [p]
        return list(dict.fromkeys(out))

    def predict_interval(
        self, Xb: pd.DataFrame, alpha: float = 0.1
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Bootstrap percentile interval per target; empty if no members were fitted."""
        if not self.members_:
            return {}
        draws = [self._predict_set(m, Xb) for m in self.members_]
        return {
            t: (
                np.quantile([d[t] for d in draws], alpha / 2, axis=0),
                np.quantile([d[t] for d in draws], 1 - alpha / 2, axis=0),
            )
            for t in self.targets
        }
