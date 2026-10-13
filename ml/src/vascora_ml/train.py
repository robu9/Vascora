"""End-to-end training pipeline: `uv run python -m vascora_ml.train` (--quick to smoke test).

Steps (all reusable from the notebooks):
1. Load + leakage guard, stratified 80/20 dev/test split (test is touched once, at the end).
2. Repeated nested CV on dev for every model x target (Optuna inside inner folds).
3. Soft-vote ensemble of the top-k models per target, scored on the same outer folds.
4. Independent vs classifier-chain (CAD -> LAD -> LCX -> RCA) comparison.
5. Re-tune chosen models on all dev data, fit calibrated ensembles + bootstrap members.
6. F1-optimal thresholds from out-of-fold dev predictions.
7. Held-out test metrics with bootstrap CIs, DeLong/McNemar vs logistic baseline.
8. SHAP global importance, artifacts and metrics.json for the API/dashboard.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

import vascora_ml
from vascora_ml import config, data, models
from vascora_ml.cv import NestedCVConfig, cv_oof_proba, outer_splits, run_nested_cv, tune
from vascora_ml.ensemble import TargetSpec, VascoraModel, parent_column
from vascora_ml.evaluation import (
    METRICS,
    best_threshold,
    bootstrap_ci,
    classification_metrics,
    curves,
    delong_test,
    mcnemar_test,
)
from vascora_ml.explain import ShapExplainer, global_importance

BASELINE = "logreg"
HIGHER_IS_BETTER = {m: m not in ("brier", "log_loss") for m in METRICS}


@dataclass
class TrainConfig:
    cv: NestedCVConfig = field(default_factory=NestedCVConfig)
    top_k: int = 3
    final_trials: int = 40
    chain_models: list[str] = field(default_factory=lambda: ["logreg", "catboost"])
    chain_min_gain: float = 0.005
    n_members: int = 10
    n_boot: int = 2000
    threshold_criterion: str = "f1"
    shap_permutations: int = 6

    @classmethod
    def quick(cls) -> "TrainConfig":
        return cls(
            cv=NestedCVConfig(outer_repeats=1, n_trials=3, models=["logreg", "rf", "lgbm"]),
            top_k=2,
            final_trials=3,
            chain_models=["logreg"],
            n_members=2,
            n_boot=200,
            shap_permutations=1,
        )


@dataclass
class Dataset:
    X_dev: pd.DataFrame
    X_test: pd.DataFrame
    Xb_dev: pd.DataFrame
    Xb_test: pd.DataFrame
    Y_dev: pd.DataFrame
    Y_test: pd.DataFrame


def prepare() -> Dataset:
    X, Y = data.load_dataset()
    X_dev, X_test, Y_dev, Y_test = data.split_dev_test(X, Y)
    return Dataset(X_dev, X_test, data.encode_base(X_dev), data.encode_base(X_test), Y_dev, Y_test)


# ---------------------------------------------------------------- CV summaries


def summarize(folds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and std of each metric across outer folds."""
    agg = folds.groupby(by)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def rank_models(summary: pd.DataFrame, target: str) -> list[str]:
    s = summary[summary.target == target].sort_values("roc_auc_mean", ascending=False)
    return s.model.tolist()


def ensemble_folds(oof: pd.DataFrame, members: dict[str, list[str]]) -> pd.DataFrame:
    """Per-fold metrics of a soft vote over each target's member models' OOF probabilities."""
    rows = []
    for target, names in members.items():
        sub = oof[(oof.target == target) & oof.model.isin(names)]
        vote = sub.groupby(["repeat", "fold", "idx"]).agg(y=("y", "first"), prob=("prob", "mean"))
        for (r, f), g in vote.groupby(level=[0, 1]):
            rows.append(
                {"target": target, "model": "ensemble", "repeat": r, "fold": f,
                 **classification_metrics(g.y.to_numpy(), g.prob.to_numpy())}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- chain experiment


def _chain_fold(
    name: str, Xb: pd.DataFrame, Y: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
    r: int, f: int, seed: int,
) -> list[dict]:
    order = list(Y.columns)
    rows = []
    a_tr, a_te = Xb.iloc[tr].copy(), Xb.iloc[te].copy()
    for t in order:
        y_tr, y_te = Y[t].to_numpy()[tr], Y[t].to_numpy()[te]
        indep = models.build(name, seed=seed).fit(Xb.iloc[tr], y_tr)
        p_ind = indep.predict_proba(Xb.iloc[te])[:, 1]
        chained = models.build(name, seed=seed).fit(a_tr, y_tr)
        p_chain = chained.predict_proba(a_te)[:, 1]
        for mode, p in (("independent", p_ind), ("chain", p_chain)):
            rows.append({"target": t, "model": name, "mode": mode, "repeat": r, "fold": f,
                         **classification_metrics(y_te, p)})
        a_tr[parent_column(t)] = cv_oof_proba(models.build(name, seed=seed), a_tr, y_tr, seed=seed)
        a_te[parent_column(t)] = p_chain
    return rows


def chain_experiment(Xb: pd.DataFrame, Y: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Same outer folds as nested CV, default hyper-parameters, paired by fold."""
    strat = Y[[c for c in Y.columns if c != "CAD"]].sum(axis=1).to_numpy()
    jobs = [
        delayed(_chain_fold)(name, Xb, Y, tr, te, r, f, cfg.cv.seed)
        for name in cfg.chain_models
        for (r, f, tr, te) in outer_splits(strat, cfg.cv)
    ]
    return pd.DataFrame(
        [row for rows in Parallel(n_jobs=cfg.cv.n_jobs)(jobs) for row in rows]
    )


def chain_decision(chain: pd.DataFrame, min_gain: float) -> dict[str, dict[str, float]]:
    """Mean paired AUC gain of chaining per target (averaged over chain base models)."""
    piv = chain.pivot_table(index=["target", "model", "repeat", "fold"], columns="mode",
                            values="roc_auc").reset_index()
    piv["gain"] = piv["chain"] - piv["independent"]
    out = {}
    for t, g in piv.groupby("target"):
        gain = float(g.gain.mean())
        out[t] = {"auc_gain": gain, "auc_gain_std": float(g.gain.std()),
                  "use_chain": bool(gain > min_gain)}
    return out


# ---------------------------------------------------------------- final model


def tune_members(
    Xb: pd.DataFrame, Y: pd.DataFrame, chosen: dict[str, list[str]], n_trials: int, cfg: TrainConfig
) -> dict[str, list[tuple[str, dict]]]:
    jobs = [
        delayed(tune)(name, Xb, Y[t].to_numpy(), n_trials, 5, cfg.cv.seed)
        for t, names in chosen.items() for name in names
    ]
    res = Parallel(n_jobs=cfg.cv.n_jobs)(jobs)
    it = iter(res)
    return {t: [(name, next(it)[0]) for name in names] for t, names in chosen.items()}


def build_specs(
    tuned: dict[str, list[tuple[str, dict]]], chain_use: dict[str, bool], order: list[str]
) -> dict[str, TargetSpec]:
    specs = {}
    for i, t in enumerate(order):
        parents = order[:i] if chain_use.get(t) else []
        specs[t] = TargetSpec(members=tuned[t], parents=parents)
    return specs


def _fold_fit(specs, Xb, Y, tr, te, seed):
    m = VascoraModel(specs, seed=seed).fit(Xb.iloc[tr], Y.iloc[tr], n_members=0, n_jobs=1)
    return te, m.predict_proba(Xb.iloc[te])


def dev_oof(specs: dict[str, TargetSpec], Xb: pd.DataFrame, Y: pd.DataFrame, cfg: TrainConfig):
    """Out-of-fold probabilities of the complete final model on dev (for thresholds)."""
    strat = Y[[c for c in Y.columns if c != "CAD"]].sum(axis=1).to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=cfg.cv.seed + 7)
    res = Parallel(n_jobs=cfg.cv.n_jobs)(
        delayed(_fold_fit)(specs, Xb, Y, tr, te, cfg.cv.seed) for tr, te in skf.split(Xb, strat)
    )
    out = pd.DataFrame(index=Xb.index, columns=list(specs), dtype=float)
    for te, probs in res:
        for t, p in probs.items():
            out.iloc[te, out.columns.get_loc(t)] = p
    return out


def evaluate_test(
    model: VascoraModel,
    baseline: dict[str, Any],
    ds: Dataset,
    cfg: TrainConfig,
    youden: dict[str, float],
) -> dict[str, Any]:
    probs = model.predict_proba(ds.Xb_test)
    out = {}
    for t in model.targets:
        y = ds.Y_test[t].to_numpy()
        p = probs[t]
        thr = model.thresholds[t]
        p_base = baseline[t].predict_proba(ds.Xb_test)[:, 1]
        out[t] = {
            "metrics": classification_metrics(y, p, thr),
            "ci95": bootstrap_ci(y, p, thr, n_boot=cfg.n_boot, seed=cfg.cv.seed),
            "metrics_at_0_5": classification_metrics(y, p, 0.5),
            "youden_threshold": youden[t],
            "metrics_at_youden": classification_metrics(y, p, youden[t]),
            "curves": curves(y, p, thr),
            "baseline": {
                "model": models.ZOO[BASELINE].label,
                "metrics": classification_metrics(y, p_base, 0.5),
                "delong": delong_test(y, p, p_base),
                "mcnemar": mcnemar_test(y, (p >= thr).astype(int), (p_base >= 0.5).astype(int)),
            },
            "n": int(len(y)),
            "prevalence": float(y.mean()),
        }
    return {"results": out, "probs": probs}


# ---------------------------------------------------------------- export


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 5)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def metric_table(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    row = summary.iloc[0]
    return {m: {"mean": row[f"{m}_mean"], "std": row[f"{m}_std"]} for m in METRICS}


def build_metrics_json(
    ds: Dataset,
    cfg: TrainConfig,
    cv_summary: pd.DataFrame,
    ens_summary: pd.DataFrame,
    chain: dict[str, dict[str, float]],
    model: VascoraModel,
    oof: pd.DataFrame,
    test: dict[str, Any],
    importance: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for t in model.targets:
        spec = model.specs[t]
        y_dev = ds.Y_dev[t].to_numpy()
        targets[t] = {
            "model": {
                "type": "calibrated soft-vote ensemble",
                "members": [{"name": n, "label": models.ZOO[n].label, "params": p}
                            for n, p in spec.members],
                "chain_parents": spec.parents,
                "calibration": "Platt (sigmoid), 5-fold out-of-fold",
                "threshold": model.thresholds[t],
                "threshold_criterion": cfg.threshold_criterion,
            },
            "test": test["results"][t],
            "dev_oof": classification_metrics(y_dev, oof[t].to_numpy(), model.thresholds[t]),
            "cv": {
                "ensemble": metric_table(ens_summary[ens_summary.target == t]),
                "models": {
                    r.model: metric_table(cv_summary[(cv_summary.target == t)
                                                     & (cv_summary.model == r.model)])
                    for r in cv_summary[cv_summary.target == t].itertuples()
                },
            },
            "chain": chain.get(t),
            "global_importance": importance[t].head(25).to_dict(orient="records")
            if t in importance else [],
        }
    return _clean({
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": vascora_ml.__version__,
        "dataset": {
            "name": "Extension of Z-Alizadeh Sani (UCI id 411)",
            "n_total": len(ds.Y_dev) + len(ds.Y_test),
            "n_dev": len(ds.Y_dev),
            "n_test": len(ds.Y_test),
            "n_features": ds.Xb_dev.shape[1],
            "excluded_from_features": sorted(data.excluded_columns()),
            "prevalence": {t: float(ds.Y_dev[t].mean()) for t in model.targets},
        },
        "validation": {
            "scheme": (
                f"{cfg.cv.outer_repeats}x repeated stratified {cfg.cv.outer_splits}-fold nested CV "
                f"(Optuna TPE, {cfg.cv.n_trials} trials, inner {cfg.cv.inner_splits}-fold "
                f"ROC-AUC) on the {len(ds.Y_dev)}-patient dev set; untouched "
                f"{len(ds.Y_test)}-patient stratified hold-out test set"
            ),
            "test_ci": f"stratified percentile bootstrap, {cfg.n_boot} resamples",
            "config": asdict(cfg),
        },
        "metric_names": METRICS,
        "targets": targets,
    })


def save_artifacts(model, explainer, metrics_json, folds, oof_cv, chain_df, ds) -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "explainer": explainer, "version": vascora_ml.__version__,
         "feature_keys": model.feature_keys_},
        config.ARTIFACTS_DIR / "vascora_model.joblib",
        compress=3,
    )
    with open(config.ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_json, fh, indent=2)
    if "params" in folds:
        folds = folds.assign(params=folds["params"].map(json.dumps))
    folds.to_csv(
        config.ARTIFACTS_DIR / "cv_folds.csv", index=False
    )
    oof_cv.to_csv(config.ARTIFACTS_DIR / "cv_oof.csv.gz", index=False)
    chain_df.to_csv(config.ARTIFACTS_DIR / "chain_experiment.csv", index=False)
    split = pd.DataFrame({"row": list(ds.Y_dev.index) + list(ds.Y_test.index),
                          "split": ["dev"] * len(ds.Y_dev) + ["test"] * len(ds.Y_test)})
    split.to_csv(config.ARTIFACTS_DIR / "split.csv", index=False)


# ---------------------------------------------------------------- main


def load_cv_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(oof_cv, folds, chain_df) saved by a previous run."""
    a = config.ARTIFACTS_DIR
    folds = pd.read_csv(a / "cv_folds.csv")
    if "params" in folds:
        folds["params"] = folds["params"].map(json.loads)
    return pd.read_csv(a / "cv_oof.csv.gz"), folds, pd.read_csv(a / "chain_experiment.csv")


def run(cfg: TrainConfig, log=print, reuse_cv: bool = False, verbose: int = 0) -> dict[str, Any]:
    """Full pipeline. `reuse_cv` loads cached nested-CV/chain results and refits the rest."""
    t0 = time.perf_counter()
    _log = log

    def log(msg: str) -> None:
        _log(f"[{time.perf_counter() - t0:6.0f}s] {msg}")

    ds = prepare()
    order = data.target_keys()
    log(f"dev={len(ds.Y_dev)} test={len(ds.Y_test)} features={ds.Xb_dev.shape[1]}")

    if reuse_cv:
        log("loading cached nested CV + chain experiment ...")
        oof_cv, folds, chain_df = load_cv_results()
    else:
        log("nested CV ...")
        oof_cv, folds = run_nested_cv(ds.Xb_dev, ds.Y_dev, cfg.cv, verbose=verbose)
    cv_summary = summarize(folds, ["target", "model"])
    chosen = {t: rank_models(cv_summary, t)[: cfg.top_k] for t in order}
    ens_folds = ensemble_folds(oof_cv, chosen)
    ens_summary = summarize(ens_folds, ["target", "model"])
    log(f"chosen members: {chosen}")

    if not reuse_cv:
        log("chain experiment ...")
        chain_df = chain_experiment(ds.Xb_dev, ds.Y_dev, cfg)
    chain = chain_decision(chain_df, cfg.chain_min_gain)
    log(f"chain: {chain}")

    log("final tuning ...")
    tuned = tune_members(ds.Xb_dev, ds.Y_dev, chosen, cfg.final_trials, cfg)
    specs = build_specs(tuned, {t: c["use_chain"] for t, c in chain.items()}, order)

    log("dev OOF thresholds ...")
    oof = dev_oof(specs, ds.Xb_dev, ds.Y_dev, cfg)
    thresholds = {
        t: best_threshold(ds.Y_dev[t].to_numpy(), oof[t].to_numpy(), cfg.threshold_criterion)
        for t in order
    }
    youden = {
        t: best_threshold(ds.Y_dev[t].to_numpy(), oof[t].to_numpy(), "youden") for t in order
    }
    log(f"thresholds ({cfg.threshold_criterion}): {thresholds}  youden: {youden}")

    log("fit final model ...")
    model = VascoraModel(specs, thresholds, seed=cfg.cv.seed).fit(
        ds.Xb_dev, ds.Y_dev, n_members=cfg.n_members, n_jobs=cfg.cv.n_jobs
    )
    baseline = {t: models.build(BASELINE).fit(ds.Xb_dev, ds.Y_dev[t]) for t in order}

    log("held-out test ...")
    test = evaluate_test(model, baseline, ds, cfg, youden)
    for t in order:
        m = test["results"][t]["metrics"]
        my = test["results"][t]["metrics_at_youden"]
        log(f"  {t}: AUC={m['roc_auc']:.3f} F1={m['f1']:.3f} acc={m['accuracy']:.3f} | "
            f"youden F1={my['f1']:.3f} acc={my['accuracy']:.3f}")

    log("SHAP ...")
    explainer = ShapExplainer(model, ds.Xb_dev)
    shap_values = explainer.explain(ds.Xb_test, cfg.shap_permutations)
    importance = {t: global_importance(ev) for t, ev in shap_values.items()}

    metrics_json = build_metrics_json(
        ds, cfg, cv_summary, ens_summary, chain, model, oof, test, importance
    )
    save_artifacts(model, explainer, metrics_json, folds, oof_cv, chain_df, ds)
    log(f"done -> {config.ARTIFACTS_DIR}")
    return {
        "ds": ds, "folds": folds, "oof_cv": oof_cv, "cv_summary": cv_summary,
        "ens_folds": ens_folds, "ens_summary": ens_summary, "chosen": chosen,
        "chain_df": chain_df, "chain": chain, "specs": specs, "oof": oof, "model": model,
        "baseline": baseline, "test": test, "explainer": explainer,
        "shap_values": shap_values, "importance": importance, "metrics": metrics_json,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="tiny config for smoke testing")
    ap.add_argument("--repeats", type=int, help="outer CV repeats")
    ap.add_argument("--trials", type=int, help="Optuna trials per inner search")
    ap.add_argument("--reuse-cv", action="store_true",
                    help="reuse cached nested-CV/chain results, refit final model only")
    args = ap.parse_args()
    cfg = TrainConfig.quick() if args.quick else TrainConfig()
    if args.repeats:
        cfg.cv.outer_repeats = args.repeats
    if args.trials:
        cfg.cv.n_trials = args.trials
    run(cfg, reuse_cv=args.reuse_cv, verbose=5)


if __name__ == "__main__":
    main()
