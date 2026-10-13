# Vascora

Interactive 3D coronary artery disease (CAD) risk visualization. Predicts overall CAD and
stenosis of the LAD, LCX and RCA from clinical data (Extension of Z-Alizadeh Sani dataset)
and maps the predictions onto a 3D heart.

> **Disclaimer:** Vascora is for decision support and educational purposes only. It is not a
> substitute for coronary angiography or formal diagnostic imaging.

Full project plan: [plan.md](plan.md).

## Status

| Part | State |
| --- | --- |
| Shared contracts (`shared/`) | Feature schema (55 inputs) and target/vessel registry |
| ML pipeline (`ml/`) | Trained and evaluated; see [results](#model-performance) |
| Backend (`backend/`) | FastAPI routes served by the trained ensemble (falls back to the heuristic placeholder if no artifact is present) |
| Frontend / 3D (`frontend/`) | Not started |

## Repository layout

```
shared/
  feature_schema.json    # input features: key, dataset column, type, unit, range, category
  vessel_registry.json   # targets (CAD, LAD, LCX, RCA) -> 3D mesh names
backend/
  app/
    main.py              # FastAPI app factory, CORS, routers under /api
    config.py            # paths and settings (env-overridable)
    registry.py          # loads shared JSON, builds the PatientFeatures model, leakage guard
    services.py          # prediction/explanation assembly
    predictors/          # Predictor protocol + heuristic stub (trained model plugs in here)
    routes/              # meta, inference, metrics, samples
    data/                # sample patient presets
  tests/
ml/
  data/                  # raw dataset (gitignored)
  artifacts/             # trained models, metrics.json
```

The target keys `CAD`, `LAD`, `LCX`, `RCA` are the single contract shared by the model
registry, API responses, `vessel_registry.json` and the 3D mesh names, which keeps
predictions and anatomy in sync.

## ML pipeline

`ml/` is a uv project on Python 3.12 (3.14 has no wheels for several of the model
libraries). It trains one calibrated ensemble per target — overall CAD plus stenosis of the
LAD, LCX and RCA — on the Extension of Z-Alizadeh Sani dataset (UCI id 411, 303 patients,
55 clinical features).

```bash
cd ml
uv sync
uv run python -m vascora_ml.train     # full run: repeated nested CV, ~2 h on 8 cores
uv run pytest                          # leakage guard, metrics, pipeline tests
```

`LAD`, `LCX`, `RCA` and `Cath` are excluded from the inputs of every model, enforced in
`vascora_ml.data.assert_no_leakage` and covered by tests. The dataset is downloaded
automatically into `ml/data/` (gitignored).

What the run does:

1. Stratified 80/20 dev/test split (stratified on the number of stenotic vessels). The test
   split is not used for training, model selection or threshold tuning.
2. Model zoo on the dev set: elastic-net logistic regression, RBF SVM, random forest,
   extra trees, XGBoost, LightGBM, CatBoost. Each is tuned with Optuna inside every fold of a
   5-fold × 3-repeat nested cross-validation, so reported CV scores are unbiased.
3. Per target, a soft vote of the top-3 models, Platt-calibrated on out-of-fold votes.
4. A classifier chain (CAD → LAD → LCX → RCA, fed out-of-fold probabilities, never true
   labels) compared against independent models on the same folds; a target joins the chain
   only when chaining lifts its CV AUC.
5. Decision thresholds maximising F1, chosen on out-of-fold dev predictions.
6. Held-out test metrics with stratified bootstrap 95% confidence intervals, plus DeLong and
   McNemar tests against the logistic-regression baseline.
7. Permutation SHAP explanations of the full ensemble in log-odds, so each attribution maps
   to exactly one clinical feature.

Artifacts land in `ml/artifacts/` (gitignored model file, committed `metrics.json`) and the
three notebooks in `ml/notebooks/` (`01_eda`, `02_training_evaluation`, `03_explainability`)
reproduce the analysis and the plots in `ml/reports/`.

### Model performance

Held-out test set (61 patients), at the F1-optimal threshold. Brackets are 95% bootstrap
confidence intervals.

| Target | ROC-AUC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| CAD | 0.846 (0.74–0.93) | 0.824 (0.74–0.90) | 0.754 | 0.854 | 0.795 |
| LAD | 0.810 (0.70–0.91) | 0.795 (0.71–0.88) | 0.754 | 0.744 | 0.853 |
| LCX | 0.755 (0.63–0.87) | 0.700 (0.58–0.81) | 0.705 | 0.636 | 0.778 |
| RCA | 0.689 (0.55–0.81) | 0.538 (0.39–0.67) | 0.607 | 0.467 | 0.636 |

Repeated nested CV on the 242-patient dev set (ensemble, mean ± SD over 15 outer folds):
CAD 0.952 ± 0.026 AUC, LAD 0.861 ± 0.034, LCX 0.711 ± 0.053, RCA 0.751 ± 0.042.

Read the CV numbers as an upper bound. The held-out split happens to be a hard one: the
logistic baseline itself scores at the 4th percentile of its own CV distribution on CAD, and
with 61 test patients the confidence intervals are wide (no model beats the baseline with
statistical significance on the test set). The test figures are the ones to quote.

Serving: the API answers `/api/predict` in about 60 ms and `/api/explain` (SHAP for all four
targets) in about 0.5 s, with repeat requests for the same patient served from cache.

## Running the backend

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 automatically).

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```

### Environment variables

| Variable | Default |
| --- | --- |
| `VASCORA_SHARED_DIR` | `<repo>/shared` |
| `VASCORA_ARTIFACTS_DIR` | `<repo>/ml/artifacts` |
| `VASCORA_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |

## API

All routes are prefixed with `/api`. Patient bodies use the feature `key`s from
`shared/feature_schema.json`; every field is optional and missing values are filled with
schema defaults (listed in `imputed` in the response). Unknown fields and out-of-range values
are rejected with `422`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Status, active model, feature and target counts |
| GET | `/schema` | Feature definitions used to build the patient form |
| GET | `/targets` | Targets and their 3D mesh mapping |
| POST | `/predict` | CAD + per-vessel probabilities, thresholds, labels |
| POST | `/explain?targets=CAD&targets=LAD` | Per-feature additive contributions (log-odds) and category totals |
| POST | `/analyze` | Prediction and explanations in one call (for live what-if updates) |
| POST | `/counterfactual?target=CAD` | Changes to modifiable features that this model scores as lower risk |
| GET | `/metrics` | Evaluation metrics from training (`available: false` until trained) |
| GET | `/samples`, `/samples/{id}` | Preset patients, hand-built (`low-risk`, `moderate-risk`, `high-risk`) and three anonymized held-out test patients (`test-low`, `test-borderline`, `test-high`) |

Example:

```bash
curl -X POST localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"age": 67, "sex": "Male", "typical_chest_pain": 1, "dm": 1, "region_rwma": 2}'
```

```json
{
  "model": {"name": "heuristic-stub", "version": "0.1.0", "is_placeholder": true},
  "cad": {"target": "CAD", "probability": 0.98, "predicted_positive": true, "threshold": 0.5, "ci": null},
  "vessels": {"LAD": {...}, "LCX": {...}, "RCA": {...}},
  "imputed": ["weight", "height", ...],
  "disclaimer": "For decision support and educational purposes only. ..."
}
```

### Placeholder model

When `ml/artifacts/vascora_model.joblib` exists, `/predict` and `/explain` are served by the
trained ensemble (`model.is_placeholder: false`), including a per-prediction 90% interval and
SHAP attributions that sum exactly to the predicted log-odds. Otherwise the API falls back to
`backend/app/predictors/stub.py`, a heuristic placeholder that is **not fitted to data** and
reports `model.is_placeholder: true`. Force the fallback with `VASCORA_PREDICTOR=stub`.

`/counterfactual` moves each schema-flagged modifiable feature into its normal range and keeps
the change only when the trained model assigns a lower probability. On the moderate-risk
preset that lowers CAD risk from 0.17 to 0.04, first by clearing hypertension, then by bringing
triglycerides down to the normal range. On the
high-risk preset it returns no hints: typical angina, age and wall-motion findings already
push the probability to ~1, and no modifiable feature moves it. A change the model does not
support is never suggested, even when it is clinically sensible.

## Leakage guard

`LAD`, `LCX`, `RCA` and `Cath` are listed in `excluded_from_features`. The registry refuses to
start if any of them (or any target column) appears as an input feature, and a test covers it.
