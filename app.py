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


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "afrihealth_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"


# ============================================================
# FEATURE SCHEMA
# ============================================================

DEFAULT_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ================================
       GLOBAL
       ================================ */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(0, 180, 180, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at top left,
                rgba(0, 120, 180, 0.08),
                transparent 28%
            ),
            #07111f;
        color: #f5f7fa;
    }

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ================================
       HEADINGS
       ================================ */

    h1, h2, h3 {
        color: #f5f7fa !important;
    }

    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2 {
        margin-top: 1.5rem;
    }

    /* ================================
       HERO
       ================================ */

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(10, 30, 50, 0.96),
                rgba(8, 38, 50, 0.94)
            );
        border: 1px solid rgba(0, 220, 200, 0.20);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(0, 220, 200, 0.10);
        color: #5fffe9;
        border: 1px solid rgba(0, 220, 200, 0.25);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        color: #ffffff;
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #b9c7d6;
        line-height: 1.7;
        max-width: 850px;
    }

    /* ================================
       CARDS
       ================================ */

    .card {
        background: rgba(13, 28, 45, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.16);
    }

    .metric-card {
        background: linear-gradient(
            135deg,
            rgba(13, 34, 52, 0.95),
            rgba(10, 28, 44, 0.95)
        );
        border: 1px solid rgba(0, 220, 200, 0.15);
        border-radius: 18px;
        padding: 1.2rem;
        min-height: 120px;
    }

    .metric-label {
        color: #8fa5b8;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        color: #5fffe9;
        font-size: 1.55rem;
        font-weight: 800;
    }

    /* ================================
       TEXT
       ================================ */

    .muted {
        color: #9fb0c0;
    }

    .small-text {
        font-size: 0.85rem;
        color: #93a5b5;
        line-height: 1.6;
    }

    .section-title {
        color: #5fffe9;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* ================================
       RESULT
       ================================ */

    .result-card {
        background:
            linear-gradient(
                135deg,
                rgba(0, 150, 130, 0.15),
                rgba(10, 30, 50, 0.95)
            );
        border: 1px solid rgba(0, 220, 200, 0.25);
        border-radius: 20px;
        padding: 1.6rem;
        margin-top: 1.2rem;
    }

    .result-title {
        color: #5fffe9;
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }

    .result-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
    }

    /* ================================
       INPUTS
       ================================ */

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
    }

    /* ================================
       BUTTONS
       ================================ */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(0, 220, 200, 0.30);
        background: linear-gradient(
            135deg,
            #0b8f87,
            #087f9a
        );
        color: white;
        font-weight: 700;
        padding: 0.7rem 1rem;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #5fffe9;
        transform: translateY(-1px);
    }

    /* ================================
       DISCLAIMER
       ================================ */

    .disclaimer {
        background: rgba(255, 193, 7, 0.07);
        border: 1px solid rgba(255, 193, 7, 0.20);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        color: #d7dce2;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 1rem 0 1.5rem 0;
    }

    /* ================================
       FOOTER
       ================================ */

    .footer {
        text-align: center;
        color: #738596;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
    }

    /* ================================
       MOBILE
       ================================ */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        .hero-subtitle {
            font-size: 1rem;
        }

        .metric-card {
            min-height: auto;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            AI + HEALTH RESEARCH PROTOTYPE
        </div>

        <div class="hero-title">
            🫀 AfriHealth AI
        </div>

        <div class="hero-subtitle">
            AI-powered health-risk intelligence for Africa.
            A research and educational prototype for exploring
            machine-learning-based cardiovascular risk classification.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESPONSIBLE AI DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>⚠️ Research and Educational Use Only</strong><br><br>

        AfriHealth AI is a machine-learning research prototype.
        It is <strong>not a medical diagnosis tool</strong> and should
        not be used to make treatment or clinical decisions.

        The underlying model was developed using the UCI Heart Disease
        dataset and has not been externally validated on a Nigerian
        or broader African population.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained machine-learning pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_data
def load_metadata():
    """Load model metadata."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH.name}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


# ============================================================
# LOAD MODEL + METADATA
# ============================================================

try:

    model = load_model()
    metadata = load_metadata()

except Exception as error:

    st.error(
        "AfriHealth AI could not load its model files."
    )

    st.exception(error)

    st.stop()


# ============================================================
# MODEL SCHEMA VALIDATION
# ============================================================

metadata_features = metadata.get(
    "input_features",
    DEFAULT_FEATURES,
)

if metadata_features != DEFAULT_FEATURES:

    st.error(
        "Model feature schema does not match the application's "
        "expected feature schema."
    )

    st.write("Expected features:", DEFAULT_FEATURES)
    st.write("Metadata features:", metadata_features)

    st.stop()


# ============================================================
# PROJECT METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Selected Model</div>
            <div class="metric-value">Logistic Regression</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    cv_auc = metadata.get(
        "cross_validation_roc_auc",
        metadata.get("cross_validation_auc"),
    )

    cv_auc_display = (
        f"{float(cv_auc):.4f}"
        if cv_auc is not None
        else "N/A"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CV ROC-AUC</div>
            <div class="metric-value">{cv_auc_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    training_records = metadata.get(
        "training_records",
        "N/A",
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Training Records</div>
            <div class="metric-value">{training_records}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:

    test_records = metadata.get(
        "test_records",
        "N/A",
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Holdout Records</div>
            <div class="metric-value">{test_records}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

st.markdown(
    '<div class="section-title">Research Input</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="small-text">
        Enter values using the coding scheme of the original UCI
        Heart Disease dataset. These inputs are intended for research
        and educational experimentation.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("risk_prediction_form"):

    st.markdown("### 👤 Demographic Information")

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
            format_func=lambda x: (
                "Female (0)"
                if x == 0
                else "Male (1)"
            ),
        )


    st.markdown("### ❤️ Clinical Measurements")

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
            format_func=lambda x: (
                "No (0)"
                if x == 0
                else "Yes (1)"
            ),
        )


    st.markdown("### 🧪 Diagnostic Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[1, 2, 3, 4],
        )

        restecg = st.selectbox(
            "Resting ECG (restecg)",
            options=[0, 1, 2],
        )

    with col2:

        exang = st.selectbox(
            "Exercise-Induced Angina (exang)",
            options=[0, 1],
            format_func=lambda x: (
                "No (0)"
                if x == 0
                else "Yes (1)"
            ),
        )

        slope = st.selectbox(
            "ST Segment Slope (slope)",
            options=[1, 2, 3],
        )

    with col3:

        ca = st.selectbox(
            "Major Vessels (ca)",
            options=[0, 1, 2, 3],
        )

        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[3, 6, 7],
        )


    submitted = st.form_submit_button(
        "🫀 Run AfriHealth AI Analysis"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [
                {
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
                }
            ],
            columns=DEFAULT_FEATURES,
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = list(model.classes_)

            if 1 in classes:

                positive_index = classes.index(1)

                probability = float(
                    probabilities[positive_index]
                )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Analysis Result</div>',
            unsafe_allow_html=True,
        )


        if int(prediction) == 1:

            result_text = "Higher-risk classification"

        else:

            result_text = "Lower-risk classification"


        probability_text = (
            f"{probability * 100:.2f}%"
            if probability is not None
            else "Not available"
        )


        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-title">
                    AfriHealth AI Classification
                </div>

                <div class="result-value">
                    {result_text}
                </div>

                <br>

                <div class="small-text">
                    Model probability for class 1:
                    <strong>{probability_text}</strong>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # Responsible interpretation
        # ----------------------------------------------------

        st.warning(
            """
            This result is a machine-learning classification output,
            not a diagnosis.

            The probability shown is the model's estimated probability
            for class 1 within this dataset and model. It should not be
            interpreted as an individual's clinical probability of having
            cardiovascular disease.

            The model has not been clinically validated for Nigerian
            or African populations.
            """
        )


        # ----------------------------------------------------
        # Display submitted data
        # ----------------------------------------------------

        with st.expander(
            "View research input"
        ):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Model Information</div>',
    unsafe_allow_html=True,
)

with st.expander(
    "🔬 View model and research details",
    expanded=False,
):

    st.write(
        "Selected model:",
        metadata.get(
            "selected_model",
            "Not recorded",
        ),
    )

    st.write(
        "Selection metric:",
        metadata.get(
            "selection_metric",
            "Not recorded",
        ),
    )

    # FIXED METADATA KEY
    st.write(
        "Cross-validation ROC-AUC:",
        metadata.get(
            "cross_validation_roc_auc",
            metadata.get(
                "cross_validation_auc",
                "Not recorded",
            ),
        ),
    )

    st.write(
        "Dataset:",
        metadata.get(
            "dataset",
            "Not recorded",
        ),
    )

    st.write(
        "Training records:",
        metadata.get(
            "training_records",
            "Not recorded",
        ),
    )

    st.write(
        "Holdout test records:",
        metadata.get(
            "test_records",
            "Not recorded",
        ),
    )

    st.write(
        "Scikit-learn version:",
        metadata.get(
            "scikit_learn_version",
            "Not recorded",
        ),
    )

    st.write(
        "Model version:",
        metadata.get(
            "version",
            "Not recorded",
        ),
    )

    st.write(
        "Created:",
        metadata.get(
            "created_at_utc",
            "Not recorded",
        ),
    )

    st.write(
        "Intended use:",
        metadata.get(
            "intended_use",
            "Research and educational use only.",
        ),
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

with st.expander(
    "📋 View input feature schema"
):

    feature_table = pd.DataFrame(
        {
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
        }
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATASET LIMITATIONS
# ============================================================

with st.expander(
    "⚠️ Research limitations"
):

    st.markdown(
        """
        **Dataset**

        The prototype uses the UCI Heart Disease dataset,
        repository ID 45.

        **Population limitation**

        The dataset is not representative of the Nigerian
        or broader African population.

        **Sample size**

        The project contains 303 records, with 242 records
        used for training and 61 records reserved as a holdout
        test set.

        **Model selection**

        Logistic Regression was selected using mean stratified
        cross-validation ROC-AUC.

        **Validation**

        The model has not undergone external validation on an
        independent Nigerian or African clinical dataset.

        **Clinical use**

        The system should not be used for diagnosis, treatment,
        patient triage, or other clinical decision-making.

        **Research direction**

        Future work should include larger and more representative
        African datasets, external validation, calibration analysis,
        fairness evaluation, clinical collaboration, privacy
        protections, and prospective validation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>AfriHealth AI</strong><br>

        AI-powered health-risk intelligence for Africa.<br><br>

        Research and educational prototype · Version 0.1.0

    </div>
    """,
    unsafe_allow_html=True,
)
