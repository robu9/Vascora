# Demonstration script (about 6 minutes)

Record the screen at 1920×1080 with the backend and frontend already running, the disclaimer not yet acknowledged, and the network panel closed. Speak to the clinical point first, then the mechanism.

## 0:00–0:40 Problem

Open the app. Read the banner: decision support only, not a substitute for angiography. On the modal, say that artery color will be a stenosis *probability*, not a picture of plaque. Click “I understand”.

Say: statistical risk scores are numbers. The point of this prototype is to put the same numbers on the three arteries a catheterization would look at — LAD, LCX and RCA — and to show which clinical inputs moved the score.

## 0:40–1:40 Input

Leave the heart in anatomy mode for a moment so the uncolored vessels are visible. In Clinical inputs, choose **High risk**: a 67-year-old man with diabetes, typical angina, T inversion and regional wall-motion abnormality.

Point at the overall CAD probability and the three vessel rows as they fill in. Switch on **Risk overlay** if it is not already on. Orbit once and use Anterior, then Left. Say that every LAD segment and the diagonal share one probability, because the dataset has no coordinates along the vessel.

## 1:40–2:40 Selection

Click the LAD on the mesh, not the button. The camera moves in and the side panel switches to LAD. Read the probability, the F1 threshold and the confidence interval. Mention that Region with RWMA is an echo input sitting in the form, and the note under Echocardiography: it is not a 3D lesion map.

Click LCX and RCA from the list so all three targets are shown.

## 2:40–4:00 Why the score moved

Scroll the explanation. Show the clinical-group bars, then the strongest factors, then the physiological table: blood pressure, lipids, ejection fraction, each with its log-odds contribution. Say plainly that a positive number raises the log-odds of the selected target and is not a percentage-point effect.

Open **What the model scores as lower risk**. On this high-risk record it may be empty. Switch to **Moderate risk**, where hypertension and triglycerides are actionable, and show a hint that the trained model actually scores as lower risk. Apply one of those edits in the form (clear hypertension or lower triglycerides) and wait for the heart color and the CAD figure to change together.

## 4:00–5:00 What-if

Change pulse rate and point at the caption: the beat is illustrative and follows the entered pulse. It is not simulated hemodynamics. Change age upward and downward so the audience sees the estimate track the form without a separate “run model” button.

## 5:00–6:00 Evidence and architecture

Open **Show metrics**. Read held-out ROC-AUC: CAD about 0.85, LAD about 0.81, LCX about 0.76, RCA about 0.69, on 61 patients. Say the cross-validation means are higher and should not be the number you quote.

Close on the path: Z-Alizadeh Sani features, leakage guard that drops LAD, LCX, RCA and Cath, calibrated ensemble, SHAP, one registry that binds those three names to the mesh. Restate the disclaimer.
