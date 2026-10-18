
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# AfriHealth AI: Research and educational prototype
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "afrihealth_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="AfriHealth AI",
    page_icon="🫀",
    layout="centered",
)


st.markdown("""
<style>
/* AfriHealth AI visual identity */
.stApp {
    background:
        radial-gradient(
            ellipse at 10% 0%,
            rgba(22, 120, 139, 0.18),
            transparent 42%
        ),
        #07111F;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

h1 {
    letter-spacing: -1.5px;
    font-weight: 800 !important;
}

h2, h3 {
    letter-spacing: -0.5px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(21, 43, 66, 0.96),
        rgba(12, 28, 44, 0.96)
    );
    border: 1px solid rgba(53, 214, 194, 0.22);
    padding: 18px;
    border-radius: 18px;
}

div[data-testid="stForm"] {
    background: rgba(16, 31, 51, 0.72);
    border: 1px solid #243650;
    padding: 22px;
    border-radius: 20px;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 46px;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    border-color: #35D6C2;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

div[data-testid="stExpander"] {
    border: 1px solid #243650;
    border-radius: 14px;
}

footer {
    visibility: hidden;
}

@media (max-width: 640px) {
    .block-container {
        padding: 1rem 0.8rem 2rem;
    }

    h1 {
        font-size: 2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Keep the original dataset's input feature names.
DEFAULT_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]

st.title("🫀 AfriHealth AI")
st.subheader("AI-powered health-risk intelligence for Africa")

st.write(
    "Explore a machine-learning research prototype that classifies "
    "cardiovascular disease status using inputs based on the UCI "
    "Heart Disease dataset."
)

st.warning(
    "**Research use only.** This application is not a medical device "
    "and must not be used to diagnose disease, estimate an individual's "
    "clinically validated risk, or guide treatment. The source dataset "
    "is small and is not representative of Nigeria's population. "
    "Consult a qualified healthcare professional about health concerns."
)

# Load model and metadata with useful error messages.
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Upload it to the repository root."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH.name}. "
            "Upload it to the repository root."
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()
except Exception as exc:
    st.error("AfriHealth AI could not load its model files.")
    st.code(str(exc))
    st.info(
        "Check that both model files are uploaded and that the "
        "deployment's package versions are compatible with the "
        "environment used to train the model."
    )
    st.stop()

# Use the metadata feature list when available.
feature_names = metadata.get("input_features", DEFAULT_FEATURES)

if not isinstance(feature_names, list) or not feature_names:
    st.error("The metadata does not contain a valid input feature list.")
    st.stop()

expected = set(DEFAULT_FEATURES)
actual = set(feature_names)

if actual != expected:
    st.error(
        "The model feature names differ from the expected UCI Heart "
        "Disease input names. Check the notebook and model metadata "
        "before accepting predictions."
    )
    st.write("Features in metadata:", feature_names)
    st.stop()

st.markdown("---")
st.header("Research input form")
st.caption(
    "Enter values using the source dataset's coding scheme. "
    "These codes are dataset values, not personalised medical advice."
)

with st.form("patient_input_form"):
    st.subheader("Basic measurements")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50,
            step=1,
            help="Age in years.",
        )
        resting_bp = st.number_input(
            "Resting blood pressure",
            min_value=50,
            max_value=250,
            value=130,
            step=1,
            help="Resting blood pressure value as recorded in the dataset.",
        )
        cholesterol = st.number_input(
            "Serum cholesterol",
            min_value=50,
            max_value=700,
            value=240,
            step=1,
            help="Cholesterol measurement in the source dataset's units.",
        )

    with col2:
        max_heart_rate = st.number_input(
            "Maximum heart rate achieved",
            min_value=40,
            max_value=250,
            value=150,
            step=1,
        )
        oldpeak = st.number_input(
            "Oldpeak",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Dataset variable related to ST depression.",
        )

    st.subheader("Dataset-coded fields")

    sex = st.selectbox(
        "Sex code",
        options=[0, 1],
        format_func=lambda x: {
            0: "0 — female in the source dataset",
            1: "1 — male in the source dataset",
        }[x],
    )

    cp = st.selectbox(
        "Chest-pain type code",
        options=[1, 2, 3, 4],
        help="Use the source dataset's original category code.",
    )

    fbs = st.selectbox(
        "Fasting blood sugar code",
        options=[0, 1],
        format_func=lambda x: {
            0: "0 — not above the dataset threshold",
            1: "1 — above the dataset threshold",
        }[x],
    )

    restecg = st.selectbox(
        "Resting ECG code",
        options=[0, 1, 2],
    )

    exang = st.selectbox(
        "Exercise-induced angina code",
        options=[0, 1],
        format_func=lambda x: {
            0: "0 — no in source dataset",
            1: "1 — yes in source dataset",
        }[x],
    )

    slope = st.selectbox(
        "ST-segment slope code",
        options=[1, 2, 3],
    )

    ca = st.selectbox(
        "Major-vessel count code (ca)",
        options=[0, 1, 2, 3],
        help="Choose a code only if you have a valid source-dataset value.",
    )

    thal = st.selectbox(
        "Thal code",
        options=[3, 6, 7],
        help="Use the source dataset's original category code.",
    )

    submitted = st.form_submit_button(
        "Run research classification",
        use_container_width=True,
    )

if submitted:
    input_values = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": resting_bp,
        "chol": cholesterol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": max_heart_rate,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    # Preserve exactly the feature order used by the training data.
    input_df = pd.DataFrame(
        [{name: input_values[name] for name in feature_names}],
        columns=feature_names,
    )

    try:
        prediction = int(model.predict(input_df)[0])

        st.markdown("---")
        st.header("Model output")

        if prediction == 1:
            st.error(
                "The model classified this input as class 1 "
                "in the source dataset."
            )
        else:
            st.success(
                "The model classified this input as class 0 "
                "in the source dataset."
            )

        st.caption(
            "Class 0 represents original target value 0. "
            "Class 1 represents original target values greater than 0. "
            "Neither output is a medical diagnosis."
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                class_1_probability = float(probabilities[classes.index(1)])

                st.metric(
                    "Model-estimated probability of dataset class 1",
                    f"{class_1_probability:.1%}",
                )

                st.caption(
                    "This is the model's probability for its dataset "
                    "class, not a clinically validated personal risk "
                    "percentage. It should not be used for medical "
                    "decisions."
                )

        with st.expander("View submitted research inputs"):
            st.dataframe(input_df, use_container_width=True)

        with st.expander("Model information"):
            st.write(
                "Selected model:",
                metadata.get("selected_model", "Not recorded"),
            )
            st.write(
                "Selection metric:",
                metadata.get("selection_metric", "Not recorded"),
            )
            st.write(
                "Cross-validation AUC:",
                metadata.get("cross_validation_auc", "Not recorded"),
            )
            st.write(
                "Dataset:",
                metadata.get("dataset", "Not recorded"),
            )
            st.write(
                "Model version:",
                metadata.get("version", "Not recorded"),
            )

    except Exception as exc:
        st.error("The model could not process these inputs.")
        st.code(str(exc))
        st.info(
            "Check the saved pipeline, input feature names, and "
            "package versions in the deployment."
        )

st.markdown("---")
st.subheader("About this project")
st.write(
    "AfriHealth AI is an educational machine-learning prototype "
    "developed to demonstrate data preprocessing, model deployment, "
    "and responsible AI communication."
)
st.write(
    "It uses the UCI Heart Disease dataset. The dataset's population, "
    "coding, sample size, and limitations mean that performance "
    "cannot be assumed to generalise to Nigerian patients."
)

st.caption(
    "AfriHealth AI | Research prototype | Not for diagnosis or treatment"
)
