"""Repeated nested cross-validation: Optuna tuning in inner folds, scoring on outer folds."""

import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from vascora_ml import models
from vascora_ml.config import SEED
from vascora_ml.evaluation import classification_metrics

Params = models.Params


@dataclass
class NestedCVConfig:
    outer_splits: int = 5
    outer_repeats: int = 3
    inner_splits: int = 3
    n_trials: int = 20
    models: list[str] = field(default_factory=lambda: list(models.ZOO))
    seed: int = SEED
    n_jobs: int = -1


def _quiet() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def inner_cv_auc(
    name: str, params: Params, X: pd.DataFrame, y: np.ndarray, splits: int, seed: int
) -> float:
    cv = StratifiedKFold(splits, shuffle=True, random_state=seed)
    scores = []
    for tr, va in cv.split(X, y):
        est = models.build(name, params, seed)
        est.fit(X.iloc[tr], y[tr])
        scores.append(roc_auc_score(y[va], est.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(scores))


def tune(
    name: str, X: pd.DataFrame, y: np.ndarray, n_trials: int, splits: int, seed: int
) -> tuple[Params, float]:
    """Optuna TPE search maximising inner-CV ROC-AUC; the defaults are always trial 0."""
    _quiet()
    spec = models.ZOO[name]
    if n_trials <= 0:
        return dict(spec.defaults), float("nan")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8)
    )
    study.enqueue_trial(spec.defaults)
    study.optimize(
        lambda t: inner_cv_auc(name, spec.space(t), X, y, splits, seed), n_trials=n_trials
    )
    return dict(study.best_params), float(study.best_value)


def _outer_task(
    target: str,
    name: str,
    repeat: int,
    fold: int,
    X: pd.DataFrame,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    cfg: NestedCVConfig,
) -> dict:
    _quiet()
    t0 = time.perf_counter()
    seed = cfg.seed + 1000 * repeat + fold
    params, inner_auc = tune(name, X.iloc[tr], y[tr], cfg.n_trials, cfg.inner_splits, seed)
    est = models.build(name, params, cfg.seed)
    est.fit(X.iloc[tr], y[tr])
    prob = est.predict_proba(X.iloc[te])[:, 1]
    return {
        "target": target,
        "model": name,
        "repeat": repeat,
        "fold": fold,
        "idx": te,
        "y": y[te],
        "prob": prob,
        "params": params,
        "inner_auc": inner_auc,
        "seconds": time.perf_counter() - t0,
    }


def outer_splits(
    y: np.ndarray, cfg: NestedCVConfig
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.outer_splits, n_repeats=cfg.outer_repeats, random_state=cfg.seed
    )
    return [
        (i // cfg.outer_splits, i % cfg.outer_splits, tr, te)
        for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y))
    ]


def run_nested_cv(
    Xb: pd.DataFrame, Y: pd.DataFrame, cfg: NestedCVConfig, verbose: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (oof, folds).

    `oof`: one row per (target, model, repeat, patient) with the out-of-fold probability.
    `folds`: one row per outer fold with tuned params and per-fold metrics.
    Splits are shared across models and targets (stratified on stenotic-vessel count) so
    paired comparisons and soft-vote ensembles of OOF predictions are valid.
    """
    strat = Y[[c for c in Y.columns if c != "CAD"]].sum(axis=1).to_numpy()
    splits = outer_splits(strat, cfg)
    jobs = [
        delayed(_outer_task)(target, name, r, f, Xb, Y[target].to_numpy(), tr, te, cfg)
        for target in Y.columns
        for name in cfg.models
        for (r, f, tr, te) in splits
    ]
    results = Parallel(n_jobs=cfg.n_jobs, verbose=verbose)(jobs)

    oof_rows, fold_rows = [], []
    for res in results:
        oof_rows.append(
            pd.DataFrame(
                {
                    "target": res["target"],
                    "model": res["model"],
                    "repeat": res["repeat"],
                    "fold": res["fold"],
                    "idx": Xb.index[res["idx"]],
                    "y": res["y"],
                    "prob": res["prob"],
                }
            )
        )
        fold_rows.append(
            {
                "target": res["target"],
                "model": res["model"],
                "repeat": res["repeat"],
                "fold": res["fold"],
                "inner_auc": res["inner_auc"],
                "seconds": res["seconds"],
                "params": res["params"],
                **classification_metrics(res["y"], res["prob"]),
            }
        )
    return pd.concat(oof_rows, ignore_index=True), pd.DataFrame(fold_rows)


def cv_oof_proba(est, X: pd.DataFrame, y: np.ndarray, splits: int = 5, seed: int = SEED):
    """Out-of-fold probabilities of a fixed estimator (no tuning)."""
    cv = StratifiedKFold(splits, shuffle=True, random_state=seed)
    out = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        m = clone(est).fit(X.iloc[tr], y[tr])
        out[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out
