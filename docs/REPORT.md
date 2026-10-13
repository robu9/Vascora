# Vascora — project report

Decision-support viewer for coronary artery disease. A calibrated ensemble scores overall CAD and stenosis of the LAD, LCX and RCA from the Extension of the Z-Alizadeh Sani dataset. Those probabilities color the matching arteries on an interactive heart. This note covers preprocessing, models, evaluation, explanations, the 3D pipeline and how to run the system.

**Disclaimer.** Vascora is for decision support and education only. It is not a substitute for coronary angiography or formal diagnostic imaging. Vessel color is a probability, not plaque location or percent narrowing.

## 1. Data and preprocessing

The source is UCI repository id 411: 303 patients and 55 clinical inputs (demographics, history, examination, vitals, labs, ECG and echocardiography), plus four labels. Overall CAD is the dataset `Cath` column. Vessel labels are LAD, LCX and RCA, each Stenotic or Normal. In this dataset, CAD means at least 50% narrowing of one or more major coronary vessels. Prevalence is about 71% CAD, 59% LAD, and 38% each for LCX and RCA.

`LAD`, `LCX`, `RCA` and `Cath` are dropped from every model matrix. `vascora_ml.data.assert_no_leakage` refuses a frame that still contains them, and the API registry refuses to start if any target column is listed as an input. Region with RWMA is kept as an echo feature. It is not treated as a map of where a lesion sits.

Binary and yes/no fields are encoded inside the training pipeline. Categorical fields (`Sex`, `BBB`, `VHD`) are one-hot encoded. The split is stratified 80/20 on the count of stenotic vessels: 242 patients for development, 61 held out. The test split is not used for training, model selection, threshold tuning or calibration. Missing inputs at request time are filled from `shared/feature_schema.json` defaults and listed as `imputed` so the dashboard can say which values were not supplied.

The same schema is the form contract. Each feature has a key, label, type, unit, range, clinical category and a modifiable flag. Adding a feature is a schema change plus a retrained artifact, not a new screen.

## 2. Model

Each target has its own calibrated soft-vote ensemble. On the development set the candidate pool is elastic-net logistic regression, RBF SVM, random forest, extra trees, XGBoost, LightGBM and CatBoost. Optuna (TPE, 20 trials, inner 3-fold ROC-AUC) runs inside every fold of a 5-fold × 3-repeat stratified nested cross-validation, so the CV scores are not the scores of a model that already saw those patients.

For each target the top three models by inner-fold AUC are refit and combined in a soft vote, then Platt-scaled on out-of-fold votes. A classifier chain CAD → LAD → LCX → RCA, fed out-of-fold probabilities rather than true labels, is compared on the same folds. A target joins the chain only when chaining lifts CV AUC by at least 0.005. Decision thresholds maximize F1 on out-of-fold development predictions. The served CAD ensemble is CatBoost, elastic-net logistic regression and extra trees, Platt-calibrated, with an F1 threshold of 0.505. Artifacts and `metrics.json` are what the API loads.

Uncertainty on a single patient is the spread across ensemble members, returned as an interval when the artifact provides one. Heart color uses the point probability only.

## 3. Evaluation

Quote the held-out test set (61 patients), not the cross-validation mean. Brackets are 95% stratified bootstrap intervals (2,000 resamples).

| Target | ROC-AUC | F1 | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| CAD | 0.846 (0.74–0.93) | 0.824 (0.74–0.90) | 0.754 | 0.854 | 0.795 |
| LAD | 0.810 (0.70–0.91) | 0.795 (0.71–0.88) | 0.754 | 0.744 | 0.853 |
| LCX | 0.755 (0.63–0.87) | 0.700 (0.58–0.81) | 0.705 | 0.636 | 0.778 |
| RCA | 0.689 (0.55–0.81) | 0.538 (0.39–0.67) | 0.607 | 0.467 | 0.636 |

Repeated nested CV on the 242-patient development set (ensemble, mean ± SD over 15 outer folds): CAD 0.952 ± 0.026 AUC, LAD 0.861 ± 0.034, LCX 0.711 ± 0.053, RCA 0.751 ± 0.042. Those numbers are an optimistic bound. The hold-out happens to be a hard slice: the logistic baseline itself lands near the bottom of its own CV distribution on CAD, and with 61 test patients the intervals are wide. DeLong and McNemar tests against that baseline are not significant on the test set. The dashboard’s Model section reads these figures from `metrics.json`, including accuracy, precision, recall, F1 and ROC-AUC.

RCA is the weakest target. The interface therefore shows a probability and a threshold, not a binary “blocked / clear” stamp, and the color ramp is continuous.

## 4. Explanations

Local explanations are permutation SHAP on the full ensemble, in log-odds, so the contributions for one patient sum with the base value to the predicted log-odds. Each attribution is one clinical feature, with its category (demographics, history, symptoms, vitals, labs, ECG, echo). The dashboard shows three views of the same vector:

- group totals, so a clinician can see whether labs, ECG or history dominate;
- the strongest individual factors, drawn as a bar;
- vitals, labs and echo measurements listed with the value, unit, an outside-range flag and the signed contribution.

Arrows and signs are direction in log-odds. They are not percentage-point effects. A separate counterfactual pass moves schema-flagged modifiable features into their normal range and keeps a hint only when this model assigns a lower probability. Weight is not an independent lever; a BMI change recomputes weight from height. On the moderate-risk preset this lowers CAD risk from about 0.17 to about 0.04. On the high-risk preset it often returns no hint, because typical angina, age and wall-motion findings already pin the probability near 1. A change the model does not support is not shown, even if it would be sensible clinically.

`/api/analyze` returns prediction and SHAP together. Repeat requests for the same patient are cached. Predict is on the order of 60 ms; a four-target explanation is on the order of half a second.

## 5. 3D pipeline and integration

The heart is an adaptation of Z-Anatomy / BodyParts3D (CC BY-SA 4.0). External atrial and ventricular surfaces, the great vessels and coronary curves are exported to a meshopt-compressed glTF (`frontend/public/models/heart.glb`, about 852 KB, 126,630 triangles, 20 meshes). LAD, LCX and RCA are each split into proximal, mid and distal segments. D1 and OM1 are representative branch assignments. The PDA is an authored right-dominant approximation because the source collection does not name a PDA. Proximal/mid/distal names are display partitions, not clinical lesion segments. There is no torso shell.

`shared/vessel_registry.json` is the only map from a prediction key to mesh names. LAD owns `LAD_prox`, `LAD_mid`, `LAD_dist` and `D1`; LCX owns its three segments and `OM1`; RCA owns its three segments and `PDA`. The viewer checks that every registry mesh exists exactly once and otherwise refuses to color the model. The same strings (`CAD`, `LAD`, `LCX`, `RCA`) are the model registry keys, the API keys and the mesh groups.

Risk mode lerps each vessel from blue (lower probability) toward amber (higher). Selection and hover raise emissive intensity. Orbit, zoom, pan, four camera presets and click-to-focus are available. Labels can be toggled. An illustrative heartbeat uses the patient’s pulse rate; it is a scale pulse, not a simulation, and it pauses for reduced-motion preferences. The scene never calls the API. It reads a Zustand store. The store is the only client of `/api/analyze` and `/api/counterfactual`.

The clinical form is generated from `/api/schema`. Choosing a preset or editing a field sends the current record, debounced by 250 ms. The response updates the CAD figure, the three vessel probabilities, the explanation for the selected target and the artery colors together. Editing weight or height recomputes BMI so those inputs cannot contradict each other.

A persistent banner states the educational limit. The first visit also requires an acknowledgment that colors are not a lesion map; the choice is remembered for the browser session.

Rendering stays on the integrated GPU path: meshopt, no texture download, antialiasing, a performance monitor that drops device pixel ratio under load, and a demand render loop whenever the heartbeat is paused.

## 6. How to run

Python 3.12 via uv for `ml/` and `backend/`. Node 22.12+ for `frontend/`. From a fresh checkout, train only if `ml/artifacts/vascora_model.joblib` is absent (`uv run python -m vascora_ml.train` in `ml/`, on the order of two hours). Otherwise:

```bash
cd backend && uv sync && uv run uvicorn app.main:app --reload --port 8000
cd frontend && npm ci && npm run dev
```

Open http://127.0.0.1:5173. Acknowledge the disclaimer, choose a sample (including three anonymized hold-out patients), then change age, blood pressure, lipids, ECG flags or the RWMA count. The heart and the side panel update together. Click LAD, LCX or RCA on the model or in the list. Open “Show metrics” for the hold-out table.

Tests: `uv run pytest` in `ml/` and `backend/`; `npm test`, `npm run build` and `npm run test:e2e` in `frontend/`. If the artifact is missing, the API falls back to an unfitted heuristic and the UI labels that state. Do not present those numbers as model output.

## 7. Limits

Three hundred and three patients, a single-center public table, and no imaging coordinates. The model cannot say where along an artery a stenosis sits, how tight it is, or whether the wall-motion abnormality matches a territory. RCA discrimination is modest. Explanations are faithful to this ensemble, not a causal account of disease. The anatomy is a representative atlas mesh, including an authored PDA, and it has not had clinical validation. Nothing in the interface should be read as a diagnosis.
