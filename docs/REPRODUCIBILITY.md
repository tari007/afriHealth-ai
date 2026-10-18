
# AfriHealth AI: Reproducibility Guide

## 1. Purpose

This document describes how to inspect, reproduce, and audit the
AfriHealth AI research prototype.

The project is intended for education and research. It is not a
clinically validated diagnostic system.

## 2. Repository Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application and dashboards |
| `afrihealth_model.joblib` | Saved model pipeline |
| `model_metadata.json` | Model selection and training metadata |
| `evaluation_predictions.csv` | Exported holdout predictions |
| `requirements.txt` | Python dependencies |
| `MODEL_CARD.md` | Intended use, risks, and limitations |

The training notebook should also be preserved as a separate research
artifact when available.

## 3. Dataset and Target

Dataset: UCI Heart Disease dataset, repository ID 45.

Recorded total: 303 records.

The target is transformed into a binary label:
- 0 when the original target is 0.
- 1 when the original target is greater than 0.

Confirm the target transformation and preprocessing against the
training notebook before reproducing results.

## 4. Model Selection

The selected model is Logistic Regression.

The recorded selection criterion is mean stratified cross-validation
ROC-AUC, approximately 0.9025.

Model selection must be based on cross-validation within the training
data. The holdout test set should remain separate from model selection,
hyperparameter tuning, and threshold selection.

## 5. Recorded Data Split

The metadata records:
- Training records: 242.
- Holdout records: 61.

For exact reproduction, record the split method, random seed,
stratification settings, preprocessing pipeline, and class distribution
from the original training notebook.

Do not assume an unspecified random seed or split procedure.

## 6. Environment

Recorded scikit-learn version: 1.6.1.

Install the dependencies listed in `requirements.txt` in an isolated
Python environment.

A saved scikit-learn pipeline may not be compatible with every future
library version. Preserve the original environment details and test
the model after any dependency upgrade.

## 7. Reproducing the Evaluation

The evaluation CSV should contain:
- `y_true`: actual holdout class.
- `y_pred`: predicted class.
- `prob_class_1`: model probability for class 1.
- `model`: selected model name.

The app calculates metrics from these exported predictions. This
checks the exported evaluation results but does not independently
recreate the original training process.

To reproduce the full experiment, rerun the training workflow using
the documented dataset, split, preprocessing, model settings, and
random seeds. Compare the resulting predictions and metrics.

## 8. Validation Checklist

Before publishing a new version:

- [ ] Confirm the dataset source and target transformation.
- [ ] Confirm no holdout records were used for model selection.
- [ ] Record the split seed and preprocessing configuration.
- [ ] Verify saved-model predictions after reloading the artifact.
- [ ] Recalculate evaluation metrics from the exported predictions.
- [ ] Check for missing, invalid, or out-of-range probabilities.
- [ ] Review class-specific performance and calibration diagnostics.
- [ ] Record the Python and dependency versions.
- [ ] Document all changes to code, data, and model artifacts.
- [ ] Preserve the limitations and intended-use statements.

## 9. External Validation

Internal cross-validation and holdout evaluation are not substitutes
for external validation.

Before claiming performance for Nigeria or other African populations,
the model requires appropriate independent data, documented
evaluation methods, relevant subgroup analysis, and review by
qualified clinical and research professionals.

Any future prospective evaluation should have suitable ethical
oversight, privacy protections, and a defined research protocol.

## 10. Reporting

When reporting results, state:
- The dataset and sample size.
- The target definition.
- The train/test split and cross-validation method.
- The model-selection criterion.
- The evaluation metrics and uncertainty.
- Whether calibration and external validation were performed.
- The limitations and intended use.

Do not describe internal test performance as clinical validation.

## 11. Responsible Use

AfriHealth AI is a research and educational prototype only. Its
outputs are not diagnoses, and they should not guide treatment
decisions.
