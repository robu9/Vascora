import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
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
    assert body["model"]["is_placeholder"] is True
    assert "not a substitute" in body["disclaimer"].lower()


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
        assert math.isclose(sum(exp["category_totals"].values()), total - exp["base_value"])


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


def test_metrics_placeholder():
    body = client.get("/api/metrics").json()
    assert "available" in body


def test_unknown_sample_404():
    assert client.get("/api/samples/nope").status_code == 404
