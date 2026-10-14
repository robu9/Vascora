import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictors import get_predictor
from app.registry import Registry, get_registry

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["targets"] == ["CAD", "LAD", "LCX", "RCA"]
    assert body["features"] == len(get_registry().features)


def test_schema_excludes_targets():
    schema = client.get("/api/schema").json()
    columns = {f["column"] for f in schema["features"]}
    assert not columns & {"LAD", "LCX", "RCA", "Cath"}


def test_registry_rejects_leaked_feature():
    registry = get_registry()
    schema = dict(registry.raw_schema)
    leaked = {**schema["features"][0], "key": "lad", "column": "LAD"}
    schema["features"] = [*schema["features"], leaked]
    with pytest.raises(ValueError, match="LAD"):
        Registry(schema, registry.raw_vessels)


def test_target_meshes_map_to_vessels():
    targets = client.get("/api/targets").json()["targets"]
    for key in ("LAD", "LCX", "RCA"):
        assert all(
            mesh.startswith(key) or mesh in {"D1", "OM1", "PDA"} for mesh in targets[key]["meshes"]
        )


def test_predict_empty_body_uses_defaults():
    res = client.post("/api/predict", json={})
    assert res.status_code == 200
    body = res.json()
    assert set(body["vessels"]) == {"LAD", "LCX", "RCA"}
    assert len(body["imputed"]) == len(get_registry().features)
    assert body["model"]["is_placeholder"] is (get_predictor().info.name == "heuristic-stub")
    assert "not a substitute" in body["disclaimer"].lower()


def test_prediction_intervals_are_ordered():
    sample = client.get("/api/samples/high-risk").json()
    body = client.post("/api/predict", json=sample["features"]).json()
    for pred in [body["cad"], *body["vessels"].values()]:
        assert 0 <= pred["probability"] <= 1
        if pred["ci"] is not None:
            lo, hi = pred["ci"]
            assert 0 <= lo <= pred["probability"] <= hi <= 1


def test_predict_rejects_invalid_input():
    assert client.post("/api/predict", json={"age": 500}).status_code == 422
    assert client.post("/api/predict", json={"sex": "X"}).status_code == 422
    assert client.post("/api/predict", json={"unknown_field": 1}).status_code == 422


def test_risk_orders_samples():
    probs = {}
    for sample in client.get("/api/samples").json():
        res = client.post("/api/predict", json=sample["features"])
        assert res.status_code == 200, res.text
        probs[sample["id"]] = res.json()["cad"]["probability"]
    assert probs["low-risk"] < probs["moderate-risk"] < probs["high-risk"]


def test_explanation_is_additive():
    sample = client.get("/api/samples/high-risk").json()
    res = client.post("/api/explain", json=sample["features"], params={"targets": ["CAD", "LAD"]})
    assert res.status_code == 200
    explanations = res.json()["explanations"]
    assert set(explanations) == {"CAD", "LAD"}
    for exp in explanations.values():
        total = exp["base_value"] + sum(c["contribution"] for c in exp["contributions"])
        assert math.isclose(total, exp["output_value"], rel_tol=1e-9)
        assert math.isclose(
            sum(exp["category_totals"].values()), total - exp["base_value"], abs_tol=1e-9
        )


def test_explanation_matches_prediction():
    sample = client.get("/api/samples/high-risk").json()
    pred = client.post("/api/predict", json=sample["features"]).json()
    exp = client.post("/api/explain", json=sample["features"]).json()["explanations"]
    assert math.isclose(exp["CAD"]["probability"], pred["cad"]["probability"], abs_tol=1e-6)
    for v in ("LAD", "LCX", "RCA"):
        assert math.isclose(
            exp[v]["probability"], pred["vessels"][v]["probability"], abs_tol=1e-6
        )


def test_explain_unknown_target():
    assert client.post("/api/explain", json={}, params={"targets": ["XYZ"]}).status_code == 422


def test_analyze_matches_predict():
    features = client.get("/api/samples/moderate-risk").json()["features"]
    analyze = client.post("/api/analyze", json=features).json()
    predict = client.post("/api/predict", json=features).json()
    assert analyze["prediction"]["cad"] == predict["cad"]
    assert math.isclose(
        analyze["explanations"]["CAD"]["probability"], predict["cad"]["probability"]
    )


def test_counterfactual_only_lowers_modifiable_features():
    sample = client.get("/api/samples/moderate-risk").json()
    res = client.post("/api/counterfactual", json=sample["features"], params={"target": "CAD"})
    assert res.status_code == 200
    body = res.json()
    modifiable = {f["key"] for f in client.get("/api/schema").json()["features"] if f["modifiable"]}
    keys = {c["key"] for c in body["changes"]}
    assert body["target"] == "CAD"
    assert body["changes"], "the moderate-risk preset is the documented demo for hints"
    assert keys <= modifiable - {"weight"}
    assert all(c["delta"] < 0 for c in body["changes"])
    assert body["combined_probability"] < min(c["probability"] for c in body["changes"])


def test_counterfactual_baseline_matches_prediction_for_chained_target():
    features = client.get("/api/samples/test-borderline").json()["features"]
    predicted = client.post("/api/predict", json=features).json()["vessels"]["LCX"]["probability"]
    body = client.post("/api/counterfactual", json=features, params={"target": "LCX"}).json()
    assert math.isclose(body["baseline_probability"], predicted, abs_tol=1e-6)


def test_cors_origins_tolerate_spaces_and_trailing_commas():
    from app.config import _origins

    assert _origins("http://a.test, http://b.test ,") == ["http://a.test", "http://b.test"]


def test_counterfactual_says_when_risk_is_already_certain():
    sample = client.get("/api/samples/high-risk").json()
    body = client.post("/api/counterfactual", json=sample["features"]).json()
    assert body["baseline_probability"] > 0.99
    assert body["changes"] == []
    assert body["note"]


def test_counterfactual_unknown_target():
    assert client.post("/api/counterfactual", json={}, params={"target": "XYZ"}).status_code == 422


def test_metrics_placeholder():
    body = client.get("/api/metrics").json()
    assert "available" in body


def test_held_out_presets_span_risk_and_need_no_imputation():
    probs = {}
    for sid in ("test-low", "test-borderline", "test-high"):
        sample = client.get(f"/api/samples/{sid}").json()
        res = client.post("/api/predict", json=sample["features"])
        assert res.status_code == 200, res.text
        assert res.json()["imputed"] == []
        probs[sid] = res.json()["cad"]["probability"]
    assert probs["test-low"] < probs["test-borderline"] < probs["test-high"]


def test_unknown_sample_404():
    assert client.get("/api/samples/nope").status_code == 404
