"""What-would-lower-risk hints: move modifiable features toward their normal range.

Only features flagged `modifiable` in the schema are touched, and a change is kept only when
the model actually assigns it a lower probability. Body size is one lever: a BMI change
recomputes weight from height, since the two are the same fact entered twice.
"""

from typing import Any

from vascora_ml.data import load_schema

# Weight is not a separate lever; it follows BMI so the two cannot contradict each other.
_COUPLED_THROUGH_BMI = "weight"
_MIN_DROP = 0.01
_MAX_STEPS = 3


def _specs() -> list[dict[str, Any]]:
    return [
        f
        for f in load_schema()["features"]
        if f.get("modifiable") and f["key"] != _COUPLED_THROUGH_BMI
    ]


def _healthier(spec: dict[str, Any], value: Any) -> Any | None:
    """Value inside the normal range closest to the current one, or the off state for flags."""
    if spec["type"] == "binary":
        return 0 if int(value) == 1 else None
    if spec["type"] == "yn":
        return "N" if value == "Y" else None
    normal = spec.get("normal")
    if spec["type"] == "number" and normal is not None:
        lo, hi = normal
        if float(value) > hi:
            return float(hi)
        if float(value) < lo:
            return float(lo)
    return None


def _apply(row: dict[str, Any], spec: dict[str, Any], new: Any) -> tuple[dict[str, Any], dict]:
    out = dict(row)
    out[spec["key"]] = new
    coupled: dict[str, Any] = {}
    if spec["key"] == "bmi":
        weight = round(float(new) * (float(row["height"]) / 100) ** 2, 1)
        out["weight"] = weight
        coupled["weight"] = weight
    return out, coupled


def _change(spec: dict[str, Any], row: dict[str, Any], new: Any, coupled: dict, prob: float,
            baseline: float) -> dict[str, Any]:
    return {
        "key": spec["key"],
        "label": spec["label"],
        "category": spec["category"],
        "unit": spec.get("unit"),
        "from_value": row[spec["key"]],
        "to_value": new,
        "coupled": coupled,
        "probability": prob,
        "delta": prob - baseline,
    }


def suggest(
    row: dict[str, Any],
    proba_fn,
    min_drop: float = _MIN_DROP,
    max_steps: int = _MAX_STEPS,
) -> dict[str, Any]:
    """Single-feature hints plus a greedy stack of up to `max_steps` of them.

    `proba_fn` maps a list of raw feature dicts to probabilities of the target.
    """
    baseline = float(proba_fn([row])[0])

    def scored(current: dict[str, Any], current_p: float, skip: set[str]):
        found = []
        for spec in _specs():
            if spec["key"] in skip:
                continue
            new = _healthier(spec, current[spec["key"]])
            if new is None:
                continue
            applied, coupled = _apply(current, spec, new)
            found.append((spec, new, coupled, applied))
        if not found:
            return []
        probs = proba_fn([a for *_, a in found])
        kept = []
        for (spec, new, coupled, _), prob in zip(found, probs, strict=True):
            if current_p - float(prob) >= min_drop:
                kept.append(_change(spec, current, new, coupled, float(prob), baseline))
        kept.sort(key=lambda c: c["probability"])
        return kept

    changes = scored(row, baseline, set())

    current, current_p, combined, used = dict(row), baseline, [], set()
    for _ in range(max_steps):
        remaining = scored(current, current_p, used)
        if not remaining:
            break
        best = remaining[0]
        current, _ = _apply(current, {"key": best["key"]}, best["to_value"])
        current_p = best["probability"]
        combined.append(best)
        used.add(best["key"])

    if changes:
        note = None
    elif baseline >= 0.9:
        note = (
            "No modifiable feature moves this probability. Non-modifiable findings "
            "already put it near certainty."
        )
    else:
        note = (
            "No modifiable feature is outside its normal range in a way that lowers predicted risk."
        )
    return {
        "baseline_probability": baseline,
        "changes": changes,
        "combined": combined,
        "combined_probability": current_p,
        "note": note,
    }
