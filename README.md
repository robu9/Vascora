# Vascora

Interactive 3D coronary artery disease (CAD) risk visualization. Predicts overall CAD and
stenosis of the LAD, LCX and RCA from clinical data (Extension of Z-Alizadeh Sani dataset)
and maps the predictions onto a 3D heart.

> **Disclaimer:** Vascora is for decision support and educational purposes only. It is not a
> substitute for coronary angiography or formal diagnostic imaging.

Full project plan: [plan.md](plan.md). Write-up: [docs/REPORT.md](docs/REPORT.md).

## Status

| Part | State |
| --- | --- |
| Shared contracts (`shared/`) | Feature schema (55 inputs) and target/vessel registry |
| ML pipeline (`ml/`) | Trained and evaluated; see [results](#model-performance) |
| Backend (`backend/`) | FastAPI routes served by the trained ensemble (falls back to the heuristic placeholder if no artifact is present) |
| Frontend / 3D (`frontend/`) | React heart viewer wired to the trained API: schema-driven inputs, live re-score, vessel colors, SHAP, counterfactuals, held-out metrics |
| Documentation (`docs/REPORT.md`) | Preprocessing, models, evaluation, 3D pipeline, usage. Demo script in `docs/DEMO_SCRIPT.md` |
| Anatomy (`assets-pipeline/`) | Z-Anatomy adaptation, editable Blender project, compressed GLB and validation scripts |

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
    predictors/          # Predictor protocol, trained ensemble adapter, heuristic stub fallback
    routes/              # meta, inference, metrics, samples
    data/                # sample patient presets
  tests/
ml/
  src/vascora_ml/        # data, features, model zoo, nested CV, ensemble, SHAP, counterfactuals
  data/                  # raw dataset (gitignored)
  artifacts/             # trained model, metrics.json, CV results
  notebooks/             # EDA, training/evaluation, explainability
  reports/               # plots produced by the notebooks
  tests/
frontend/
  src/                   # React app: store, heart scene, patient form, clinical panels
  public/models/         # heart.glb plus attribution and source license
  e2e/                   # Playwright browser tests
assets-pipeline/         # Blender export, meshopt compression and validation of heart.glb
docs/                    # REPORT.md and DEMO_SCRIPT.md
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

Artifacts land in `ml/artifacts/` (model file and `metrics.json` are committed; the raw dataset stays gitignored) and the
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

Serving: the API answers `/api/predict` in about 0.2 s, most of it spent scoring the ten
bootstrap refits behind each 90% interval, and `/api/explain` (SHAP for all four targets) in
about 0.5 s, with repeat explanations for the same patient served from cache.

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

## Running the heart viewer

Requires Node 22.12+ and npm. Start the backend above, then in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. Acknowledge the decision-support disclaimer. The viewer
starts in anatomy mode, which shows a realistic textured heart. Choose a sample
patient, or edit any clinical input: the form is built from `/api/schema`, and
changes re-score CAD, LAD, LCX and RCA. When the prediction returns, the viewer
switches to the risk overlay: the wall each artery typically supplies (LAD front,
LCX left side, RCA right side and underside) takes that artery's probability color,
and colors and figures update together.
The factor breakdown, grouped contributions and lower-risk hints fill in after that.
Select an artery by clicking its territory
on the heart or in the results panel to read its probability, confidence
interval, SHAP contributions and the vitals, labs and echo measurements that
fed the estimate. “Show metrics” loads the held-out evaluation. Use the four
camera presets, orbit/zoom/pan, territory labels (risk overlay), and the heartbeat
pause control.
Reduced-motion preferences pause the heartbeat. Pulse rate in the form sets the
illustrative beat.

The frontend proxies `/api` to port 8000 locally. For a separately hosted API,
copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE_URL` to its
origin (without `/api`); configure backend `VASCORA_CORS_ORIGINS` for that frontend.
The production build is in `frontend/dist`; deployment must supply the API origin
at build time or reverse-proxy `/api` to the backend.

```bash
cd frontend
npm test                  # mapping and asynchronous state tests
npm run build             # TypeScript and production build
npm run asset:validate    # names, materials, compression and budgets
npx playwright install chromium
npm run test:e2e           # starts local frontend/backend if needed
```

The viewer presents **representative educational anatomy** and the trained
ensemble's probabilities. Territory colors do not show lesion position,
narrowing severity, flow speed, or myocardial dysfunction. The illustrative
heartbeat uses `pr` (75 bpm by default); it is not a physiological simulation.

The viewer draws neshallads' Realistic Human Heart (CC BY 4.0). It is a single
textured mesh, so the risk overlay colors it by territory: a shader tints the
ventricular wall by the direction each registry `territory` (`anterior`, `lateral`,
`inferior`) faces, and clicks map back to the same territories. `heart.glb`, the
Z-Anatomy / BodyParts3D adaptation (CC BY-SA 4.0), is loaded to check the
vessel-name contract and to size the realistic heart; it is not drawn. D1 and OM1 are representative branch assignments; the PDA is an authored
right-dominant approximation because the source did not identify a PDA explicitly.
Anatomy has not received clinical validation. See
[asset reproduction and limitations](assets-pipeline/README.md),
[credits and asset license](frontend/public/models/ATTRIBUTION.md), and
[source provenance](assets-pipeline/provenance.json). `LICENSE-source.txt` is the
unchanged Z-Anatomy notice for that coronary adaptation.

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
| GET | `/metrics` | Held-out and cross-validated evaluation from training |
| GET | `/samples`, `/samples/{id}` | Preset patients, hand-built (`low-risk`, `moderate-risk`, `high-risk`) and three anonymized held-out test patients (`test-low`, `test-borderline`, `test-high`) |

Example:

```bash
curl -X POST localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"age": 67, "sex": "Male", "typical_chest_pain": 1, "dm": 1, "region_rwma": 2}'
```

```json
{
  "model": {"name": "calibrated-ensemble", "version": "1.0.0", "is_placeholder": false},
  "cad": {"target": "CAD", "probability": 0.997, "predicted_positive": true, "threshold": 0.505, "ci": [0.988, 0.999]},
  "vessels": {"LAD": {...}, "LCX": {...}, "RCA": {...}},
  "imputed": ["weight", "height", ...],
  "disclaimer": "For decision support and educational purposes only. ..."
}
```

### Placeholder model

When `ml/artifacts/vascora_model.joblib` exists, `/predict` and `/explain` are served by the
trained ensemble (`model.is_placeholder: false`), including a per-prediction 90% interval (the
spread of ten bootstrap refits of the whole model) and SHAP attributions that sum exactly to the predicted log-odds. Otherwise the API falls back to
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
