
# AfriHealth AI: External Validation Research Protocol

## 1. Working Title

Evaluation of a Logistic Regression-Based Heart Disease
Classification Model: Internal Performance, Explainability,
and a Framework for External Validation.

## 2. Background

Machine-learning models can support research into cardiovascular
classification. However, good performance on one dataset does not
guarantee reliable performance on other populations.

AfriHealth AI is an educational and research prototype that uses
the UCI Heart Disease dataset. The project combines binary
classification, model evaluation, and coefficient-based explanations.

An external validation study is needed to investigate whether the
existing model retains its performance on a genuinely independent
dataset with compatible features and outcome definitions.

## 3. Research Problem

The principal research problem is whether the observed internal
performance of the AfriHealth AI model generalises to independent
data.

The current development dataset does not establish performance
for Nigeria or other African populations.

## 4. Aim

To evaluate the reproducibility, statistical performance, and
potential transportability of the existing AfriHealth AI model
using a prespecified validation procedure.

## 5. Research Objectives

1. Document the original model, dataset, target definition, and
   preprocessing pipeline.
2. Reproduce the recorded internal evaluation results.
3. Evaluate the frozen model on a suitable independent dataset.
4. Estimate discrimination, class-specific performance, and
   probability accuracy with uncertainty intervals.
5. Examine data quality, distribution differences, and subgroup
   performance where the sample size permits.
6. Document limitations, ethical safeguards, and implications
   for future research.

## 6. Research Questions

1. How does the saved model perform on an independent dataset?
2. How precise are its estimated performance metrics?
3. Does its probability output agree with observed outcome
   frequencies in the validation sample?
4. Which differences between the development and validation
   datasets may explain performance changes?
5. What further evidence would be required before considering
   any clinical application?

## 7. Study Design

A retrospective external validation study is a possible design
when a suitable independent dataset is available and its use
is authorised.

The design, setting, sampling period, eligibility criteria,
data source, and reference outcome must be specified before
analysis begins.

No external validation should be claimed until the independent
dataset and evaluation have actually been completed.

## 8. Development Dataset

The existing model was developed using the UCI Heart Disease
dataset, repository ID 45.

Recorded dataset size: 303 records.
Recorded training split: 242 records.
Recorded holdout split: 61 records.

The binary target is class 0 for original target value 0 and
class 1 for original target values greater than 0.

These details must be checked against the original training
notebook before publication.

## 9. Existing Model

The selected model is Logistic Regression. Model selection used
mean stratified cross-validation ROC-AUC, recorded as approximately
0.9025.

The external validation procedure must use the frozen saved model
and its original preprocessing pipeline. No fitting, threshold
selection, or model selection should use the external evaluation
data.

## 10. Validation Dataset

Before evaluation, document:

- Dataset name and source.
- Study setting and geographic location.
- Collection period.
- Eligibility criteria and sampling procedure.
- Number of records and outcome prevalence.
- Feature definitions, units, and category coding.
- Missing data and exclusions.
- Outcome definition and label quality.
- Independence from model development.
- Data access permissions and ethical approvals, where required.

## 11. Statistical Analysis Plan

Primary analyses should include ROC-AUC and its confidence interval.

Secondary analyses may include sensitivity, specificity, accuracy,
balanced accuracy, precision, F1 score, Brier score, and calibration
plots.

Report confusion matrix counts and the number of positive and
negative records.

Confidence intervals should be calculated using a prespecified
method. The resampling method, number of replicates, and limitations
should be documented.

Any subgroup analysis should be prespecified where possible and
reported cautiously when subgroup samples are small.

## 12. Missing Data and Data Quality

Describe the amount and pattern of missingness, data exclusions,
implausible values, duplicate records, and any differences in
measurement or coding.

Do not silently alter values to make the model accept the dataset.
Any conversion, exclusion, or imputation must be documented.

## 13. Ethics and Data Governance

Confirm data ownership, permission, privacy protections, and
applicable institutional requirements before using any dataset.

If patient-level data are obtained from a Nigerian hospital or
other clinical setting, obtain the required institutional and
research ethics review before accessing or analysing them.

Use de-identified data and apply appropriate security and access
controls.

## 14. Limitations

Potential limitations include small sample size, selection bias,
differences in feature measurement, outcome-label differences,
uncertain calibration, and limited representativeness.

The current model has not been established as clinically valid
for Nigeria or other African populations.

## 15. Reporting and Dissemination

Report the actual methods and results, including unsuccessful
validation, rather than only favourable metrics.

Follow applicable reporting guidance for prediction-model
evaluation, such as TRIPOD+AI, and assess risk of bias using
appropriate guidance such as PROBAST+AI.

Do not describe internal evaluation as external validation.
Do not present model output as a diagnosis or treatment recommendation.

## 16. Protocol Status

Status: Draft for methodological review.

The protocol must be finalised before the external validation
analysis. Study-specific details must be completed from verified
sources, not assumed or invented.
