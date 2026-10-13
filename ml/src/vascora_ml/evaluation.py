"""Classification metrics, bootstrap CIs, DeLong / McNemar tests and curve exports."""

from typing import Any

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
]


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def best_threshold(y: np.ndarray, p: np.ndarray, criterion: str = "f1") -> float:
    """Threshold maximising F1 (or Youden's J) on out-of-fold probabilities."""
    y = np.asarray(y).astype(int)
    grid = np.linspace(0.05, 0.95, 181)
    if criterion == "youden":
        scores = [recall_score(y, p >= t) + _specificity(y, p >= t) - 1 for t in grid]
    else:
        scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    scores = np.asarray(scores)
    # Middle of the plateau of best thresholds is more stable than the first one.
    best = np.flatnonzero(np.isclose(scores, scores.max()))
    return float(grid[best[len(best) // 2]])


def _specificity(y: np.ndarray, pred: np.ndarray) -> float:
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    return tn / (tn + fp) if (tn + fp) else 0.0


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs for every metric (resamples patients, stratified by class)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.hstack([rng.choice(pos, (n_boot, len(pos))), rng.choice(neg, (n_boot, len(neg)))])
    samples = _batch_metrics(y[idx], p[idx], threshold)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        m: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for m, v in samples.items()
    }


def _batch_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Vectorised `classification_metrics` over rows of (n_boot, n) arrays."""
    pred = p >= threshold
    yb = y.astype(bool)
    tp = (pred & yb).sum(1).astype(float)
    fp = (pred & ~yb).sum(1).astype(float)
    fn = (~pred & yb).sum(1).astype(float)
    tn = (~pred & ~yb).sum(1).astype(float)
    n_pos, n_neg = tp + fn, tn + fp

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)

    prec, rec, spec = div(tp, tp + fp), div(tp, n_pos), div(tn, n_neg)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # Rank-based AUC (Mann-Whitney) with average ranks for ties.
    ranks = np.apply_along_axis(stats.rankdata, 1, p)
    auc = ((ranks * yb).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # Average precision: mean precision at each positive in descending score order.
    order = np.argsort(-p, axis=1, kind="stable")
    ys = np.take_along_axis(yb, order, 1)
    cum_tp = np.cumsum(ys, 1)
    prec_at_k = cum_tp / np.arange(1, ys.shape[1] + 1)
    ap = (prec_at_k * ys).sum(1) / n_pos
    return {
        "accuracy": (tp + tn) / y.shape[1],
        "balanced_accuracy": (rec + spec) / 2,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "mcc": div(tp * tn - fp * fn, mcc_den),
        "roc_auc": auc,
        "pr_auc": ap,
        "brier": ((p - y) ** 2).mean(1),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(1),
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict[str, float]:
    """DeLong test for two correlated ROC AUCs (Sun & Xu fast implementation)."""
    y = np.asarray(y).astype(int)
    preds = np.vstack([p_a, p_b])
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([_midrank(r) for r in pos])
    ty = np.vstack([_midrank(r) for r in neg])
    tz = np.vstack([_midrank(r) for r in preds[:, np.r_[np.flatnonzero(y == 1),
                                                         np.flatnonzero(y == 0)]]])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired correct/incorrect predictions."""
    y = np.asarray(y).astype(int)
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def curves(y: np.ndarray, p: np.ndarray, threshold: float, n_points: int = 60) -> dict[str, Any]:
    """Down-sampled ROC/PR/calibration curves and confusion matrix for the dashboard."""
    y = np.asarray(y).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    def thin(a, b):
        if len(a) <= n_points:
            return [[round(float(x), 4), round(float(z), 4)] for x, z in zip(a, b, strict=True)]
        idx = np.unique(np.linspace(0, len(a) - 1, n_points).astype(int))
        return [[round(float(a[i]), 4), round(float(b[i]), 4)] for i in idx]

    return {
        "roc": thin(fpr, tpr),
        "pr": thin(rec[::-1], prec[::-1]),
        "calibration": thin(mean_pred, frac_pos),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
