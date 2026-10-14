import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from vascora_ml import data
from vascora_ml.evaluation import (
    _batch_metrics,
    best_threshold,
    classification_metrics,
    delong_test,
)
from vascora_ml.features import FeatureEngineer


@pytest.fixture(scope="module")
def dataset():
    return data.load_dataset()


def test_no_label_columns_in_features(dataset):
    X, Y = dataset
    banned = {"LAD", "LCX", "RCA", "Cath"}
    assert banned <= data.excluded_columns()
    assert not banned & set(X.columns)
    assert not banned & {f.column for f in data.features()}
    assert list(Y.columns) == ["CAD", "LAD", "LCX", "RCA"]


@pytest.mark.parametrize("col", ["LAD", "LCX", "RCA", "Cath"])
def test_leakage_guard_rejects_labels(col):
    with pytest.raises(ValueError, match="leakage"):
        data.assert_no_leakage(["age", col])


def test_labels_and_shape(dataset):
    X, Y = dataset
    assert X.shape == (303, 55)
    assert Y.isin([0, 1]).all().all()
    assert Y.CAD.sum() == 216
    assert (Y[["LAD", "LCX", "RCA"]].sum(axis=1) > 0).sum() == 217


def test_encode_base_is_numeric_and_decodable(dataset):
    X, _ = dataset
    Xb = data.encode_base(X)
    assert list(Xb.columns) == data.feature_keys()
    assert not Xb.isna().any().any()
    row = X.iloc[0]
    for key in ["sex", "bbb", "vhd", "obesity", "region_rwma", "age"]:
        assert data.decode_value(key, Xb.iloc[0][key]) == row[key]


def test_split_is_disjoint_and_stratified(dataset):
    X, Y = dataset
    X_dev, X_test, Y_dev, Y_test = data.split_dev_test(X, Y)
    assert not set(X_dev.index) & set(X_test.index)
    assert len(X_test) == 61
    assert abs(Y_dev.CAD.mean() - Y_test.CAD.mean()) < 0.05


def test_feature_engineer_passes_extra_columns(dataset):
    X, _ = dataset
    Xb = data.encode_base(X)
    Xb["p_CAD"] = np.linspace(0, 1, len(Xb))
    fe = FeatureEngineer().fit(Xb)
    out = fe.transform(Xb)
    names = list(fe.get_feature_names_out())
    assert out.shape == (len(Xb), len(names))
    assert "p_CAD" in names and "nlr" in names and "bbb" not in names
    assert "exertional_cp" not in names  # constant in this dataset
    np.testing.assert_allclose(out[:, names.index("p_CAD")], Xb["p_CAD"])


def test_metrics_and_delong():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p_good = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    p_bad = rng.uniform(size=200)
    m = classification_metrics(y, p_good)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p_good))
    d = delong_test(y, p_good, p_bad)
    assert d["auc_a"] == pytest.approx(roc_auc_score(y, p_good))
    assert d["auc_b"] == pytest.approx(roc_auc_score(y, p_bad))
    assert d["p_value"] < 1e-6
    assert 0.05 <= best_threshold(y, p_good) <= 0.95


def test_batch_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, (3, 80))
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, (3, 80)), 1e-6, 1 - 1e-6)
    batch = _batch_metrics(y, p, 0.45)
    for i in range(3):
        ref = classification_metrics(y[i], p[i], 0.45)
        for k, v in ref.items():
            assert batch[k][i] == pytest.approx(v, abs=1e-9), k


@pytest.mark.parametrize("fast, stock", [
    ("FastRandomForestClassifier", "RandomForestClassifier"),
    ("FastExtraTreesClassifier", "ExtraTreesClassifier"),
])
def test_fast_forest_matches_stock(dataset, fast, stock):
    import sklearn.ensemble

    from vascora_ml import models

    X, Y = dataset
    Xb = data.encode_base(X).to_numpy()
    a = getattr(models, fast)(n_estimators=50, random_state=0).fit(Xb, Y.CAD)
    b = getattr(sklearn.ensemble, stock)(n_estimators=50, random_state=0).fit(Xb, Y.CAD)
    np.testing.assert_allclose(a.predict_proba(Xb), b.predict_proba(Xb), atol=1e-12)


def test_predict_row_matches_predict_proba(dataset):
    from vascora_ml.ensemble import TargetSpec, VascoraModel

    X, Y = dataset
    Xb = data.encode_base(X).iloc[:80]
    Ys = Y.loc[Xb.index]
    specs = {
        "CAD": TargetSpec([("logreg", {"C": 0.1, "l1_ratio": 0.5})]),
        "LAD": TargetSpec([("logreg", {"C": 0.1, "l1_ratio": 0.5})], parents=["CAD"]),
    }
    model = VascoraModel(specs).fit(Xb, Ys[list(specs)])
    row = Xb.iloc[0].to_dict()
    a = model.predict_row(row)
    b = model.predict_proba(Xb.iloc[[0]])
    for t in specs:
        assert a[t] == pytest.approx(float(b[t][0]), abs=1e-9)


def test_row_interval_matches_bootstrap_interval(dataset):
    from vascora_ml.ensemble import TargetSpec, VascoraModel

    X, Y = dataset
    Xb = data.encode_base(X).iloc[:80]
    Ys = Y.loc[Xb.index]
    specs = {
        "CAD": TargetSpec([("logreg", {"C": 0.1, "l1_ratio": 0.5})]),
        "LAD": TargetSpec([("logreg", {"C": 0.1, "l1_ratio": 0.5})], parents=["CAD"]),
    }
    model = VascoraModel(specs).fit(Xb, Ys[list(specs)], n_members=4, n_jobs=1)
    row = Xb.iloc[0].to_dict()
    row_iv = model.predict_row_interval(row)
    lo_b, hi_b = zip(*(model.predict_interval(Xb.iloc[[0]])[t] for t in specs), strict=True)
    for i, t in enumerate(specs):
        p, lo, hi = row_iv[t]
        assert lo <= p <= hi
        assert lo == pytest.approx(min(float(lo_b[i][0]), p), abs=1e-9)
        assert hi == pytest.approx(max(float(hi_b[i][0]), p), abs=1e-9)


def test_encode_rejects_unknown_category(dataset):
    X, _ = dataset
    bad = X.head(2).copy()
    bad["vhd"] = ["N", "Huge"]
    with pytest.raises(ValueError, match="unknown categories"):
        data.encode_base(pd.DataFrame(bad))
