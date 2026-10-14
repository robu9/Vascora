"""Candidate model zoo: pipeline builders, default params and Optuna search spaces."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from vascora_ml.config import SEED
from vascora_ml.features import FeatureEngineer

Params = dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    build: Callable[[Params, int], Any]
    space: Callable[[optuna.Trial], Params]
    defaults: Params


def _pipe(estimator, scale: bool) -> Pipeline:
    steps = [("features", FeatureEngineer())]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _logreg(p: Params, seed: int):
    return _pipe(
        LogisticRegression(
            C=p["C"], l1_ratio=p["l1_ratio"], solver="saga", max_iter=5000, random_state=seed
        ),
        scale=True,
    )


def _logreg_space(t: optuna.Trial) -> Params:
    return {
        "C": t.suggest_float("C", 1e-3, 10.0, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
    }


def _svm(p: Params, seed: int):
    # Platt-scale the SVC margin with an inner CV via CalibratedClassifierCV.
    svc = SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", random_state=seed)
    return _pipe(CalibratedClassifierCV(svc, method="sigmoid", cv=3), scale=True)


def _svm_space(t: optuna.Trial) -> Params:
    return {
        "C": t.suggest_float("C", 1e-2, 100.0, log=True),
        "gamma": t.suggest_float("gamma", 1e-4, 1e-1, log=True),
    }


def _forest_space(t: optuna.Trial) -> Params:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 400, step=100),
        "max_depth": t.suggest_int("max_depth", 3, 14),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "max_features": t.suggest_float("max_features", 0.1, 0.7),
    }


class _DirectTreeAverage:
    """Same result as the stock forest `predict_proba`, without per-tree joblib dispatch.

    The stock path costs ~0.2 ms of overhead per tree, which dominates single-patient
    predictions and SHAP evaluations.
    """

    def predict_proba(self, X):
        X = self._validate_X_predict(X)
        total = self.estimators_[0].predict_proba(X, check_input=False)
        for tree in self.estimators_[1:]:
            total += tree.predict_proba(X, check_input=False)
        return total / len(self.estimators_)


class FastRandomForestClassifier(_DirectTreeAverage, RandomForestClassifier):
    pass


class FastExtraTreesClassifier(_DirectTreeAverage, ExtraTreesClassifier):
    pass


def _rf(p: Params, seed: int):
    return _pipe(FastRandomForestClassifier(**p, n_jobs=1, random_state=seed), scale=False)


def _et(p: Params, seed: int):
    return _pipe(FastExtraTreesClassifier(**p, n_jobs=1, random_state=seed), scale=False)


def _xgb(p: Params, seed: int):
    return _pipe(
        XGBClassifier(**p, n_jobs=1, random_state=seed, eval_metric="logloss", verbosity=0),
        scale=False,
    )


def _xgb_space(t: optuna.Trial) -> Params:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def _lgbm(p: Params, seed: int):
    return _pipe(
        LGBMClassifier(**p, subsample_freq=1, n_jobs=1, random_state=seed, verbosity=-1),
        scale=False,
    )


def _lgbm_space(t: optuna.Trial) -> Params:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 4, 31),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 40),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _cat(p: Params, seed: int):
    return _pipe(
        CatBoostClassifier(
            **p, thread_count=1, random_seed=seed, verbose=0, allow_writing_files=False
        ),
        scale=False,
    )


def _cat_space(t: optuna.Trial) -> Params:
    return {
        "iterations": t.suggest_int("iterations", 100, 500, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": t.suggest_int("depth", 2, 6),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": t.suggest_float("random_strength", 0.1, 5.0, log=True),
    }


_FOREST_DEFAULTS: Params = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "max_features": 0.3,
}

ZOO: dict[str, ModelSpec] = {
    s.name: s
    for s in [
        ModelSpec("logreg", "Elastic-net Logistic Regression", _logreg, _logreg_space,
                  {"C": 0.1, "l1_ratio": 0.5}),
        ModelSpec("svm", "SVM (RBF, Platt)", _svm, _svm_space, {"C": 1.0, "gamma": 0.01}),
        ModelSpec("rf", "Random Forest", _rf, _forest_space, dict(_FOREST_DEFAULTS)),
        ModelSpec("et", "Extra Trees", _et, _forest_space, dict(_FOREST_DEFAULTS)),
        ModelSpec("xgb", "XGBoost", _xgb, _xgb_space,
                  {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
                   "min_child_weight": 2.0, "subsample": 0.8, "colsample_bytree": 0.6,
                   "reg_lambda": 1.0, "reg_alpha": 0.01}),
        ModelSpec("lgbm", "LightGBM", _lgbm, _lgbm_space,
                  {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 8,
                   "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.6,
                   "reg_lambda": 1.0}),
        ModelSpec("catboost", "CatBoost", _cat, _cat_space,
                  {"iterations": 300, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3.0,
                   "random_strength": 1.0}),
    ]
}


def build(name: str, params: Params | None = None, seed: int = SEED):
    spec = ZOO[name]
    return spec.build(params if params is not None else spec.defaults, seed)
