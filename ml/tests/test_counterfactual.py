import numpy as np
import pytest

from vascora_ml.counterfactual import suggest
from vascora_ml.data import load_schema


def _row(**overrides):
    row = {f["key"]: f["default"] for f in load_schema()["features"]}
    row.update(overrides)
    return row


def _proba(rows):
    """Drops only for clearing diabetes, blood pressure and BMI. Ignores everything else."""
    out = []
    for r in rows:
        p = 0.8
        p -= 0.20 * (r["dm"] == 0)
        p -= 0.10 * (r["bp"] <= 130)
        p -= 0.15 * (r["bmi"] <= 25)
        out.append(p)
    return np.array(out, dtype=float)


def test_only_model_backed_modifiable_changes():
    row = _row(dm=1, bp=160, bmi=32, height=170, hdl=30, fbs=180, typical_chest_pain=1)
    result = suggest(row, _proba)

    keys = {c["key"] for c in result["changes"]}
    assert keys == {"dm", "bp", "bmi"}
    assert "typical_chest_pain" not in keys
    assert "hdl" not in keys  # healthier on paper, but this model does not agree
    assert all(c["delta"] <= -0.01 for c in result["changes"])

    bmi = next(c for c in result["changes"] if c["key"] == "bmi")
    assert bmi["to_value"] == 25
    assert bmi["coupled"]["weight"] == round(25 * (170 / 100) ** 2, 1)


def test_greedy_stack_compounds():
    row = _row(dm=1, bp=160, bmi=32, height=170)
    result = suggest(row, _proba)

    assert [c["key"] for c in result["combined"]] == ["dm", "bmi", "bp"]
    assert result["combined_probability"] == pytest.approx(result["baseline_probability"] - 0.45)
    assert result["combined_probability"] < min(c["probability"] for c in result["changes"])


def test_nothing_to_change_when_already_in_range():
    result = suggest(_row(bmi=24, ldl=90), _proba)
    assert result["changes"] == []
    assert result["combined"] == []
    assert result["combined_probability"] == result["baseline_probability"]
