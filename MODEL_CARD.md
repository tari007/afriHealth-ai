
# AfriHealth AI: Model Card

## 1. Overview

AfriHealth AI is a research and educational prototype that
demonstrates binary classification using a heart disease dataset.

The project includes a prediction interface, evaluation metrics,
probability calibration diagnostics, and coefficient-based model
explanations.

**Tagline:** AI-powered health-risk intelligence for Africa.

## 2. Intended Use

The system is intended for:
- Machine-learning education and demonstrations.
- Academic experimentation with classification pipelines.
- Exploring model evaluation and explainability.
- Developing reproducible research workflows.

It is not intended for:
- Diagnosing heart disease.
- Recommending treatment or medication.
- Replacing a qualified healthcare professional.
- Making decisions about an individual's clinical care.
- Establishing cardiovascular risk in Nigerian or other African
  populations without appropriate external validation.

## 3. Model Details

- Selected model: Logistic Regression.
- Model type: supervised binary classification.
- Model artifact: `afrihealth_model.joblib`.
- Metadata: `model_metadata.json`.
- Evaluation export: `evaluation_predictions.csv`.
- Training framework: scikit-learn.
- Recorded scikit-learn version: 1.6.1.
- Model selection criterion: mean stratified cross-validation
  ROC-AUC.

The recorded mean cross-validation ROC-AUC is approximately 0.9025.
This is an internal model-selection result, not evidence of clinical
validity.

## 4. Dataset

The project uses the UCI Heart Disease dataset, repository ID 45.

Recorded dataset size: 303 records.
Recorded training split: 242 records.
Recorded holdout test split: 61 records.

The binary target is defined as:
- Class 0: original target equals 0.
- Class 1: original target is greater than 0.

The data do not establish representative performance for Nigeria
or Africa. Dataset provenance, preprocessing, exclusions, and
the exact split procedure should be checked against the training
notebook before formal publication.

## 5. Input Features

The recorded model inputs are:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

Feature codes and units should be interpreted according to the
dataset documentation and the implemented preprocessing pipeline.

## 6. Evaluation

Model selection used mean stratified cross-validation ROC-AUC.

The project also reports holdout metrics such as accuracy,
precision, recall, F1 score, ROC-AUC, balanced accuracy, sensitivity,
specificity, and Brier score.

The holdout contains only 61 records. Metrics can vary substantially
with a different split or evaluation sample.

The holdout metrics must not be described as independent external
validation. Do not compare models using the holdout results as though
it were an untouched test set if those results influenced model choice.

## 7. Explainability

The application displays Logistic Regression coefficients and
feature contributions for individual inputs.

These explanations describe how transformed features influence
the model's output. They do not demonstrate causation or establish
medical importance.

Coefficient magnitude can be affected by feature scaling, encoding,
correlations, and the model specification.

## 8. Limitations and Risks

- The dataset is small by modern clinical prediction standards.
- Representativeness for Nigeria and Africa has not been established.
- Independent external validation has not been established.
- Probability calibration is not established by a single calibration
  curve or Brier score.
- Missing values, measurement practices, and dataset biases may affect
  model behaviour.
- Model explanations may be misunderstood as causal or clinical advice.
- Performance in deployment may differ from performance in the
  development dataset.

## 9. Privacy

Do not enter identifiable patient information into a public demo.
Do not upload identifiable health data to the repository or include
such data in research reports.

## 10. Responsible Use

This is a research and educational screening-support prototype only.
It is not a diagnostic device and must not be used to make treatment
decisions.

## 11. Versioning

Project version: 0.1.0 (recorded in metadata).

For each release, record the code revision, model artifact version,
dataset version, preprocessing steps, split seed, dependency versions,
and evaluation results.
