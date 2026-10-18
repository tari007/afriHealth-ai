
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AfriHealth AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "afrihealth_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

DEFAULT_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]


# ============================================================
# LOAD MODEL AND METADATA
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH.name}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing metadata file: {METADATA_PATH.name}"
        )
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    metadata = load_metadata()
except Exception as error:
    st.error("AfriHealth AI could not load its model files.")
    st.code(str(error))
    st.stop()


if metadata.get("input_features", DEFAULT_FEATURES) != DEFAULT_FEATURES:
    st.error(
        "The feature list in model_metadata.json does not match "
        "the features expected by app.py."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🫀 AfriHealth AI")
st.subheader("AI-powered health-risk intelligence for Africa")

st.write(
    "A research and educational prototype for exploring "
    "machine-learning-based cardiovascular risk classification."
)

st.info(
    "**Research and educational use only.** This prototype is "
    "not a medical diagnostic tool. Its model uses the UCI Heart "
    "Disease dataset and has not been externally validated on "
    "Nigerian or broader African populations."
)

st.divider()


# ============================================================
# PROJECT METRICS
# ============================================================

st.header("📊 Project Overview")

selected_model = metadata.get("selected_model", "Not recorded")

cv_auc = metadata.get(
    "cross_validation_roc_auc",
    metadata.get("cross_validation_auc"),
)

if cv_auc is not None:
    try:
        cv_auc_display = f"{float(cv_auc):.4f}"
    except (ValueError, TypeError):
        cv_auc_display = str(cv_auc)
else:
    cv_auc_display = "Not recorded"

training_records = metadata.get("training_records", "N/A")
test_records = metadata.get("test_records", "N/A")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Selected Model",
        value=str(selected_model),
    )

with col2:
    st.metric(
        label="Cross-validation ROC-AUC",
        value=cv_auc_display,
    )

col3, col4 = st.columns(2)

with col3:
    st.metric(
        label="Training Records",
        value=str(training_records),
    )

with col4:
    st.metric(
        label="Holdout Records",
        value=str(test_records),
    )

st.caption(
    "The ROC-AUC above is a cross-validation model-selection metric. "
    "It is not an individual's probability of disease or a measure "
    "of guaranteed clinical performance."
)

st.divider()


# ============================================================
# RESEARCH INPUT FORM
# ============================================================

st.header("🩺 Research Input")

st.write(
    "Enter values using the coding scheme of the original "
    "UCI Heart Disease dataset."
)

with st.form("risk_prediction_form"):

    st.subheader("Demographic Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            step=1,
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda value: (
                "Female (0)" if value == 0 else "Male (1)"
            ),
        )

    st.subheader("Clinical Measurements")

    col1, col2 = st.columns(2)

    with col1:
        trestbps = st.number_input(
            "Resting Blood Pressure (trestbps)",
            min_value=50,
            max_value=250,
            value=120,
            step=1,
        )

        chol = st.number_input(
            "Serum Cholesterol (chol)",
            min_value=50,
            max_value=700,
            value=200,
            step=1,
        )

        thalach = st.number_input(
            "Maximum Heart Rate (thalach)",
            min_value=50,
            max_value=250,
            value=150,
            step=1,
        )

    with col2:
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda value: (
                "No (0)" if value == 0 else "Yes (1)"
            ),
        )

    st.subheader("Diagnostic Features")

    col1, col2 = st.columns(2)

    with col1:
        cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        exang = st.selectbox(
            "Exercise-Induced Angina (exang)",
            options=[0, 1],
            format_func=lambda value: (
                "No (0)" if value == 0 else "Yes (1)"
            ),
        )

    with col2:
        slope = st.selectbox("ST Segment Slope (slope)", [1, 2, 3])
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [3, 6, 7])

    submitted = st.form_submit_button(
        "🫀 Run AfriHealth AI Analysis",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    input_data = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }],
        columns=DEFAULT_FEATURES,
    )

    try:
        prediction = int(model.predict(input_data)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(
                    probabilities[classes.index(1)]
                )

    except Exception as error:
        st.error("The model could not process these inputs.")
        st.code(str(error))
        st.stop()

    st.divider()
    st.header("📋 Analysis Result")

    if prediction == 1:
        st.warning("Model classification: Class 1")
    else:
        st.success("Model classification: Class 0")

    if probability is not None:
        st.metric(
            label="Model-estimated probability for class 1",
            value=f"{probability * 100:.2f}%",
        )
    else:
        st.write("Probability output is not available.")

    st.warning(
        "This is a model classification, not a diagnosis. "
        "The model-estimated probability is not a clinically "
        "validated estimate of an individual's disease risk. "
        "Do not use this output to make medical or treatment decisions."
    )

    with st.expander("View submitted research inputs"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()
st.header("🔬 Model Information")

with st.expander("View model and research details"):

    st.write("Selected model:", selected_model)

    st.write(
        "Selection metric:",
        metadata.get("selection_metric", "Not recorded"),
    )

    # Correct metadata key, with backward compatibility.
    st.write(
        "Cross-validation ROC-AUC:",
        cv_auc_display,
    )

    st.write(
        "Dataset:",
        metadata.get("dataset", "Not recorded"),
    )

    st.write(
        "Training records:",
        training_records,
    )

    st.write(
        "Holdout test records:",
        test_records,
    )

    st.write(
        "Scikit-learn version:",
        metadata.get("scikit_learn_version", "Not recorded"),
    )

    st.write(
        "Model version:",
        metadata.get("version", "Not recorded"),
    )

    st.write(
        "Created (UTC):",
        metadata.get("created_at_utc", "Not recorded"),
    )

    st.write(
        "Intended use:",
        metadata.get(
            "intended_use",
            "Research and educational use only.",
        ),
    )


# ============================================================
# FEATURE SCHEMA
# ============================================================

with st.expander("📚 View input feature schema"):

    feature_table = pd.DataFrame({
        "Feature": DEFAULT_FEATURES,
        "Purpose": [
            "Age",
            "Sex coding",
            "Chest pain type",
            "Resting blood pressure",
            "Serum cholesterol",
            "Fasting blood sugar",
            "Resting ECG",
            "Maximum heart rate",
            "Exercise-induced angina",
            "ST depression",
            "ST segment slope",
            "Major vessels",
            "Thalassemia",
        ],
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RESEARCH LIMITATIONS
# ============================================================

with st.expander("⚠️ Research limitations"):

    st.markdown(
        """
        **Dataset:** UCI Heart Disease dataset, repository ID 45.

        **Sample size:** 303 records, with 242 used for training
        and 61 reserved for holdout testing.

        **Model selection:** Logistic Regression was selected using
        mean stratified cross-validation ROC-AUC.

        **Population limitation:** The dataset is not representative
        of the Nigerian or broader African population.

        **External validation:** The model has not been externally
        validated on an independent Nigerian or African clinical dataset.

        **Clinical use:** Do not use this prototype for diagnosis,
        treatment, patient triage, or clinical decision-making.

        **Future research:** Larger representative datasets, external
        validation, calibration, fairness evaluation, privacy safeguards,
        and clinical collaboration are needed.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AfriHealth AI | AI-powered health-risk intelligence for Africa | "
    "Research and educational prototype | Version 0.1.0"
)
