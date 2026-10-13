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
| Backend (`backend/`) | FastAPI routes working, served by a **heuristic placeholder model** |
| ML pipeline (`ml/`) | Not started; trained models will replace the placeholder |
| Frontend / 3D (`frontend/`) | React heart viewer, sample patients, vessel selection and API risk overlay |
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

Open http://127.0.0.1:5173. The viewer starts in anatomy mode. Choose one of the
three sample patients to load vessel probabilities; select LAD, LCX or RCA on the
model or in the results panel. Use the four camera presets, orbit/zoom/pan, labels,
and the heartbeat pause control. Reduced-motion preferences pause the heartbeat.

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

The viewer presents **representative educational anatomy** and the current
**heuristic placeholder** estimates. Vessel colors do not show lesion position,
narrowing severity, flow speed, or myocardial dysfunction. The illustrative
heartbeat uses `pr` (75 bpm by default); it is not a physiological simulation.

The model is adapted from Z-Anatomy / BodyParts3D. D1 and OM1 are representative
branch assignments; the PDA is an authored right-dominant approximation because
the source did not identify a PDA explicitly. Anatomy has not received clinical
validation. See [asset reproduction and limitations](assets-pipeline/README.md),
[credits and asset license](frontend/public/models/ATTRIBUTION.md), and
[source provenance](assets-pipeline/provenance.json). Adapted model assets retain
CC BY-SA 4.0 licensing. The source notice is preserved with the assets.

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
| GET | `/metrics` | Evaluation metrics from training (`available: false` until trained) |
| GET | `/samples`, `/samples/{id}` | Preset patients (`low-risk`, `moderate-risk`, `high-risk`) |

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

Until the ML pipeline is trained, `/predict` and `/explain` are served by
`backend/app/predictors/stub.py`: hand-set log-odds weights on known CAD risk factors. It is
**not fitted to data** and every response reports `model.is_placeholder: true`. Its
explanations use the same additive format SHAP will return, so the frontend can be built
against the final contract now.

## Leakage guard

`LAD`, `LCX`, `RCA` and `Cath` are listed in `excluded_from_features`. The registry refuses to
start if any of them (or any target column) appears as an input feature, and a test covers it.
