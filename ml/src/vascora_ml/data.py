"""Dataset loading, schema handling and the leakage guard."""

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vascora_ml import config


@dataclass(frozen=True)
class Feature:
    key: str
    column: str
    label: str
    type: str
    category: str
    options: tuple[Any, ...] | None
    default: Any


@dataclass(frozen=True)
class Target:
    key: str
    column: str
    positive: str
    kind: str


@cache
def load_schema() -> dict[str, Any]:
    with open(config.SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@cache
def load_registry() -> dict[str, Any]:
    with open(config.REGISTRY_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@cache
def features() -> tuple[Feature, ...]:
    return tuple(
        Feature(
            key=f["key"],
            column=f["column"],
            label=f["label"],
            type=f["type"],
            category=f["category"],
            options=tuple(f["options"]) if "options" in f else None,
            default=f["default"],
        )
        for f in load_schema()["features"]
    )


@cache
def targets() -> tuple[Target, ...]:
    return tuple(
        Target(key=k, column=t["column"], positive=t["positive"], kind=t["kind"])
        for k, t in load_registry()["targets"].items()
    )


def feature_keys() -> list[str]:
    return [f.key for f in features()]


def target_keys() -> list[str]:
    return [t.key for t in targets()]


def excluded_columns() -> set[str]:
    return set(load_schema()["excluded_from_features"]) | {t.column for t in targets()}


def assert_no_leakage(columns: list[str]) -> None:
    """Raise if any label column (LAD, LCX, RCA, Cath) or its schema key is a model input."""
    banned = excluded_columns()
    banned_keys = {f.key for f in features() if f.column in banned}
    leaked = (set(columns) & banned) | (set(columns) & banned_keys)
    if leaked:
        raise ValueError(f"Target leakage: label columns used as features: {sorted(leaked)}")


def dataset_path() -> Path:
    return config.DATA_DIR / config.DATASET_FILE


def download_dataset(force: bool = False) -> Path:
    path = dataset_path()
    if path.exists() and not force:
        return path
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.DATASET_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
        path.write_bytes(zf.read(name))
    return path


def load_raw() -> pd.DataFrame:
    return pd.read_excel(download_dataset())


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, Y): X keyed by schema feature key with raw values, Y as 0/1 per target key."""
    raw = load_raw()
    cols = {f.column: f.key for f in features()}
    missing = set(cols) - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing schema columns: {sorted(missing)}")
    X = raw[list(cols)].rename(columns=cols)
    assert_no_leakage(list(X.columns))
    Y = pd.DataFrame(
        {t.key: (raw[t.column].astype(str) == t.positive).astype(int) for t in targets()}
    )
    for t in targets():
        if Y[t.key].nunique() != 2:
            raise ValueError(f"Target {t.key}: positive value {t.positive!r} not found")
    return X, Y


def encode_base(X: pd.DataFrame) -> pd.DataFrame:
    """Map raw schema values to one numeric column per feature key (schema order).

    yn -> 0/1, categorical -> option index, ordinal/number/binary -> float. This is the
    space SHAP explains in, so each attribution maps to exactly one clinical feature.
    """
    out = {}
    for f in features():
        col = X[f.key]
        if f.type == "yn":
            out[f.key] = (col == "Y").astype(float)
        elif f.type == "categorical":
            assert f.options is not None
            lookup = {opt: i for i, opt in enumerate(f.options)}
            mapped = col.map(lookup)
            if mapped.isna().any():
                bad = sorted(col[mapped.isna()].astype(str).unique())
                raise ValueError(f"{f.key}: unknown categories {bad}")
            out[f.key] = mapped.astype(float)
        else:
            out[f.key] = pd.to_numeric(col).astype(float)
    return pd.DataFrame(out, index=X.index)


def decode_value(key: str, code: float) -> Any:
    """Inverse of `encode_base` for one feature (used to show readable SHAP values)."""
    f = next(f for f in features() if f.key == key)
    if f.type == "yn":
        return "Y" if code >= 0.5 else "N"
    if f.type == "categorical":
        assert f.options is not None
        return f.options[int(round(code))]
    if f.type in ("binary", "ordinal"):
        return int(round(code))
    return float(code)


def stratification_labels(Y: pd.DataFrame) -> np.ndarray:
    """Number of stenotic vessels (0-3); CAD is ~equivalent to >=1, so this balances all four."""
    vessels = [t.key for t in targets() if t.kind == "vessel"]
    return Y[vessels].sum(axis=1).to_numpy()


def split_dev_test(
    X: pd.DataFrame, Y: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_dev, X_test, Y_dev, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed, stratify=stratification_labels(Y)
    )
    return X_dev, X_test, Y_dev, Y_test
