
# AfriHealth AI

### AI-Powered Health-Risk Intelligence for Africa

AfriHealth AI is a research and educational prototype that demonstrates
machine-learning classification using the UCI Heart Disease dataset.

## Project objectives

- Demonstrate a complete machine-learning workflow.
- Apply preprocessing through a reusable model pipeline.
- Package and deploy a trained classification model.
- Communicate model outputs and limitations responsibly.

## Technology stack

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- GitHub
- Streamlit Community Cloud

## Dataset

The project uses the UCI Heart Disease dataset.

The dataset contains 303 records and is relatively small. It is not
representative of Nigeria's population. Results should not be assumed
to generalise to Nigerian patients or other populations.

## Model development

Several classification models were evaluated using stratified
cross-validation. The saved pipeline includes preprocessing and
prediction steps.

The selected model and evaluation metadata are stored in
`model_metadata.json`.

## Running locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Responsible AI and limitations

AfriHealth AI is a research and educational demonstration only.

- It is not a medical device.
- It does not provide a diagnosis.
- Its class probability is not a clinically validated personal risk.
- It has not been externally validated for Nigerian patients.
- It must not be used to make treatment or healthcare decisions.

People with health concerns should consult qualified healthcare
professionals.

## Project files

- `app.py`: Streamlit application
- `requirements.txt`: Python dependencies
- `afrihealth_model.joblib`: trained preprocessing and model pipeline
- `model_metadata.json`: model metadata

## Future development

Potential research directions include external validation, evaluation
on suitable local data with appropriate approvals, model calibration,
explainability, fairness assessment, and secure deployment practices.

## Author

Developed as an AI and machine-learning research portfolio project.
