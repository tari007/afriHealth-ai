import io
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# OPTIONAL PDF SUPPORT
# ============================================================

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    REPORTLAB_AVAILABLE = True

except ImportError:
    REPORTLAB_AVAILABLE = False


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AfriHealth AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_FILE = "afrihealth_model.joblib"
METADATA_FILE = "model_metadata.json"
EVALUATION_FILE = "evaluation_predictions.csv"

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

EXPECTED_FEATURE_RANGES = {
    "age": (0, 120),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (50, 300),
    "chol": (50, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (50, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}

MODEL_NAME = "AfriHealth AI"
APP_VERSION = "Build 06A / Research Validation"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #5f6b7a;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background: rgba(128, 128, 128, 0.04);
        margin-bottom: 0.5rem;
    }

    .research-note {
        padding: 1rem;
        border-left: 5px solid #0f766e;
        background: rgba(15, 118, 110, 0.07);
        border-radius: 5px;
        margin: 1rem 0;
    }

    .warning-note {
        padding: 1rem;
        border-left: 5px solid #d97706;
        background: rgba(217, 119, 6, 0.08);
        border-radius: 5px;
        margin: 1rem 0;
    }

    .danger-note {
        padding: 1rem;
        border-left: 5px solid #dc2626;
        background: rgba(220, 38, 38, 0.08);
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}

    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_evaluation_data():
    if not os.path.exists(EVALUATION_FILE):
        return None

    try:
        return pd.read_csv(EVALUATION_FILE)
    except Exception:
        return None


def get_probability_for_class_1(model, X):
    """
    Return probability of target class 1.

    For a standard binary classifier:
        classes_ = [0, 1]
        probability column 1 corresponds to class 1.

    The function also handles models whose classes are ordered
    differently.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not provide predict_proba().")

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise ValueError("Unexpected probability output shape.")

    classes = getattr(model, "classes_", None)

    if classes is None:
        if probabilities.shape[1] != 2:
            raise ValueError("Binary class probabilities could not be identified.")
        return probabilities[:, 1]

    classes = np.asarray(classes)

    matches = np.where(classes == 1)[0]

    if len(matches) != 1:
        raise ValueError(
            f"Could not identify class 1 in model classes: {classes.tolist()}"
        )

    return probabilities[:, matches[0]]


def get_binary_predictions(model, X):
    """
    Obtain predictions and normalize them to binary 0/1.
    """

    predictions = model.predict(X)
    predictions = np.asarray(predictions)

    unique_values = np.unique(predictions)

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            "The model returned non-binary predictions: "
            f"{unique_values.tolist()}"
        )

    return predictions.astype(int)


def calculate_metrics(y_true, y_pred, probabilities):
    """
    Calculate all major binary classification metrics from exactly
    the same y_true, y_pred and probability arrays.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "Specificity": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "Brier Score": brier_score_loss(y_true, probabilities),
    }

    return metrics


def bootstrap_metric_ci(
    y_true,
    y_pred,
    probabilities,
    metric_name,
    n_bootstrap=1000,
    random_state=42,
):
    """
    Class-stratified bootstrap confidence interval.

    Each bootstrap sample preserves the original number of Class 0
    and Class 1 records.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    class_0_indices = np.where(y_true == 0)[0]
    class_1_indices = np.where(y_true == 1)[0]

    if len(class_0_indices) == 0 or len(class_1_indices) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)

    values = []

    for _ in range(n_bootstrap):

        sample_0 = rng.choice(
            class_0_indices,
            size=len(class_0_indices),
            replace=True,
        )

        sample_1 = rng.choice(
            class_1_indices,
            size=len(class_1_indices),
            replace=True,
        )

        sample_indices = np.concatenate([sample_0, sample_1])

        y_b = y_true[sample_indices]
        pred_b = y_pred[sample_indices]
        prob_b = probabilities[sample_indices]

        try:
            if metric_name == "Accuracy":
                value = accuracy_score(y_b, pred_b)

            elif metric_name == "Balanced Accuracy":
                value = balanced_accuracy_score(y_b, pred_b)

            elif metric_name == "Sensitivity":
                value = recall_score(
                    y_b,
                    pred_b,
                    pos_label=1,
                    zero_division=0,
                )

            elif metric_name == "Specificity":
                value = recall_score(
                    y_b,
                    pred_b,
                    pos_label=0,
                    zero_division=0,
                )

            elif metric_name == "Precision":
                value = precision_score(
                    y_b,
                    pred_b,
                    pos_label=1,
                    zero_division=0,
                )

            elif metric_name == "F1 Score":
                value = f1_score(
                    y_b,
                    pred_b,
                    pos_label=1,
                    zero_division=0,
                )

            elif metric_name == "ROC-AUC":
                value = roc_auc_score(y_b, prob_b)

            elif metric_name == "Brier Score":
                value = brier_score_loss(y_b, prob_b)

            else:
                continue

            if np.isfinite(value):
                values.append(value)

        except Exception:
            continue

    if len(values) < 100:
        return np.nan, np.nan

    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)

    return float(lower), float(upper)


def create_bootstrap_results(
    y_true,
    y_pred,
    probabilities,
    n_bootstrap=1000,
):
    """
    Generate a complete metric table containing estimate,
    lower 95% CI and upper 95% CI.
    """

    estimates = calculate_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    rows = []

    for metric_name, estimate in estimates.items():

        lower, upper = bootstrap_metric_ci(
            y_true,
            y_pred,
            probabilities,
            metric_name,
            n_bootstrap=n_bootstrap,
        )

        rows.append(
            {
                "Metric": metric_name,
                "Estimate": estimate,
                "95% CI Lower": lower,
                "95% CI Upper": upper,
            }
        )

    return pd.DataFrame(rows)


def create_roc_figure(y_true, probabilities, title):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
    )

    auc_value = roc_auc_score(
        y_true,
        probabilities,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC curve (AUC = {auc_value:.3f})",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(
                dash="dash",
                width=1,
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        template="plotly_white",
    )

    return fig


def create_calibration_figure(y_true, probabilities, title):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode="lines+markers",
            name="AfriHealth AI",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(
                dash="dash",
                width=1,
            ),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed proportion",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        template="plotly_white",
    )

    return fig


def interpret_calibration(y_true, probabilities):
    """
    Simple descriptive interpretation.

    This is deliberately not presented as a clinical calibration
    assessment.
    """

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="uniform",
    )

    if len(fraction_of_positives) < 2:
        return (
            "There are not enough populated probability bins to provide "
            "a meaningful descriptive calibration interpretation."
        )

    differences = fraction_of_positives - mean_predicted_value

    mean_absolute_difference = float(
        np.mean(np.abs(differences))
    )

    if mean_absolute_difference < 0.05:
        interpretation = "The calibration curve is relatively close to the ideal diagonal."
    elif mean_absolute_difference < 0.10:
        interpretation = "The calibration curve shows moderate deviation from the ideal diagonal."
    else:
        interpretation = "The calibration curve shows substantial deviation from the ideal diagonal."

    return (
        f"{interpretation} The average absolute difference between "
        f"observed proportions and mean predicted probabilities across "
        f"populated bins is approximately "
        f"{mean_absolute_difference:.3f}. This is a descriptive assessment "
        f"and does not establish clinical calibration."
    )


def format_metric(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.3f}"


def format_ci(lower, upper):
    if pd.isna(lower) or pd.isna(upper):
        return "N/A"

    return f"{lower:.3f}–{upper:.3f}"


def make_metrics_download(metrics_df):
    return metrics_df.to_csv(index=False).encode("utf-8")


def make_external_predictions_download(
    external_df,
    predictions,
    probabilities,
):
    output = external_df.copy()

    output["predicted_class"] = predictions
    output["predicted_probability_class_1"] = probabilities

    return output.to_csv(index=False).encode("utf-8")


def create_pdf_report(
    source,
    n_records,
    class_0,
    class_1,
    metrics_df,
    calibration_text,
    constant_features,
):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=7,
    )

    story = []

    story.append(
        Paragraph(
            "AfriHealth AI External Validation Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Research use only.</b> This report describes model "
            "performance on an external dataset. It does not establish "
            "clinical safety, diagnostic validity, or suitability for "
            "individual patient care.",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "1. Dataset Information",
            heading_style,
        )
    )

    dataset_table = Table(
        [
            ["Dataset source", source],
            ["Records evaluated", str(n_records)],
            ["Class 0", str(class_0)],
            ["Class 1", str(class_1)],
        ],
        colWidths=[150, 350],
    )

    dataset_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]
        )
    )

    story.append(dataset_table)

    story.append(
        Paragraph(
            "2. Performance Results",
            heading_style,
        )
    )

    table_data = [
        [
            "Metric",
            "Estimate",
            "95% CI",
        ]
    ]

    for _, row in metrics_df.iterrows():
        table_data.append(
            [
                row["Metric"],
                f"{row['Estimate']:.3f}",
                format_ci(
                    row["95% CI Lower"],
                    row["95% CI Upper"],
                ),
            ]
        )

    metrics_table = Table(
        table_data,
        colWidths=[180, 100, 150],
    )

    metrics_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )

    story.append(metrics_table)

    story.append(
        Paragraph(
            "3. Calibration",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            calibration_text,
            body_style,
        )
    )

    story.append(
        Paragraph(
            "4. Data Quality Notes",
            heading_style,
        )
    )

    if constant_features:
        story.append(
            Paragraph(
                "Constant features detected in the external dataset: "
                + ", ".join(constant_features)
                + ". These features had no variation in the evaluated "
                "cohort and should be considered when interpreting "
                "generalisability.",
                body_style,
            )
        )
    else:
        story.append(
            Paragraph(
                "No constant features were detected in the external dataset.",
                body_style,
            )
        )

    story.append(
        Paragraph(
            "5. Research Limitations",
            heading_style,
        )
    )

    limitations = [
        "External validation does not establish clinical validity.",
        "The model was not trained or validated specifically on a Nigerian population.",
        "Dataset shift may affect model performance across countries, hospitals and patient populations.",
        "Confidence intervals describe statistical uncertainty within this evaluation sample and do not remove population-level uncertainty.",
        "Predicted probabilities should not be interpreted as calibrated individual clinical risk without appropriate clinical validation.",
        "Prospective and appropriately governed Nigerian validation is required before clinical deployment.",
    ]

    for limitation in limitations:
        story.append(
            Paragraph(
                "• " + limitation,
                body_style,
            )
        )

    story.append(
        Paragraph(
            "6. Recommended Next Research Step",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "Conduct a locally relevant validation study using appropriately "
            "governed Nigerian clinical data. The study should define the "
            "target population, outcome definition, inclusion and exclusion "
            "criteria, preprocessing protocol, missing-data strategy, "
            "threshold selection, calibration assessment and statistical "
            "analysis plan before evaluation.",
            body_style,
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()


def get_pipeline_components(model):
    """
    Return preprocessing and estimator where possible.
    """

    if hasattr(model, "named_steps"):
        steps = model.named_steps

        if len(steps) >= 2:
            names = list(steps.keys())

            preprocessing = steps[names[-2]]
            estimator = steps[names[-1]]

            return preprocessing, estimator

    return None, model


def get_feature_names_from_preprocessor(preprocessing, original_features):
    """
    Attempt to obtain transformed feature names.
    """

    if preprocessing is None:
        return original_features

    if hasattr(preprocessing, "get_feature_names_out"):
        try:
            names = preprocessing.get_feature_names_out(
                original_features
            )
            return list(names)
        except Exception:
            pass

    return original_features


def get_logistic_coefficients(model):
    """
    Return logistic regression coefficient information if available.
    """

    preprocessing, estimator = get_pipeline_components(model)

    if not hasattr(estimator, "coef_"):
        return None

    coefficients = np.asarray(estimator.coef_)

    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        return None

    coefficients = coefficients[0]

    classes = getattr(estimator, "classes_", np.array([0, 1]))

    if len(classes) != 2:
        return None

    # sklearn binary logistic regression coef_ represents classes_[1].
    # If classes_[1] is 1, coefficients already represent Class 1.
    # Otherwise reverse the sign so the interpretation is always
    # "effect on Class 1".
    if classes[1] == 1:
        class_1_coefficients = coefficients
    elif classes[0] == 1:
        class_1_coefficients = -coefficients
    else:
        return None

    names = get_feature_names_from_preprocessor(
        preprocessing,
        DEFAULT_FEATURES,
    )

    if len(names) != len(class_1_coefficients):
        names = [
            f"feature_{i + 1}"
            for i in range(len(class_1_coefficients))
        ]

    return pd.DataFrame(
        {
            "Feature": names,
            "Coefficient": class_1_coefficients,
            "Absolute Effect": np.abs(class_1_coefficients),
        }
    ).sort_values(
        "Absolute Effect",
        ascending=False,
    )


def get_local_contributions(model, input_df):
    """
    Calculate approximate local linear contributions for a logistic
    regression pipeline.

    Contribution = transformed feature value × coefficient.

    This is appropriate for linear logistic regression models.
    """

    preprocessing, estimator = get_pipeline_components(model)

    if not hasattr(estimator, "coef_"):
        return None

    try:
        transformed = preprocessing.transform(input_df)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        coefficients = np.asarray(estimator.coef_)[0]

        classes = getattr(
            estimator,
            "classes_",
            np.array([0, 1]),
        )

        if classes[1] == 1:
            class_1_coefficients = coefficients
        elif classes[0] == 1:
            class_1_coefficients = -coefficients
        else:
            return None

        if transformed.shape[1] != len(class_1_coefficients):
            return None

        feature_names = get_feature_names_from_preprocessor(
            preprocessing,
            DEFAULT_FEATURES,
        )

        contributions = transformed[0] * class_1_coefficients

        result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Contribution": contributions,
                "Absolute Contribution": np.abs(contributions),
            }
        )

        return result.sort_values(
            "Absolute Contribution",
            ascending=False,
        )

    except Exception:
        return None


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = load_model()
metadata = load_metadata()
evaluation_data = load_evaluation_data()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 AfriHealth AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Research-focused heart disease prediction, evaluation and explainability"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="research-note">
    <b>Research and educational use only.</b><br>
    AfriHealth AI is a research prototype. It is not a medical device,
    does not provide a diagnosis, and must not be used as a substitute
    for qualified medical assessment or treatment.
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Important: The current model was not specifically trained or "
    "clinically validated on African or Nigerian populations. "
    "Predicted probabilities are not established individual clinical risk."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("AfriHealth AI")

st.sidebar.markdown(
    """
    ### Research Modules

    1. Prediction
    2. Model Evaluation
    3. Reliability Analysis
    4. Explainability
    5. External Validation
    6. Research Information
    """
)

st.sidebar.caption(
    f"{MODEL_NAME} · {APP_VERSION}"
)


# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.error(
        f"Model file `{MODEL_FILE}` was not found. "
        "Ensure the model file is in the repository root."
    )

    st.stop()


# ============================================================
# PROJECT METRICS
# ============================================================

st.subheader("Project Overview")

selected_model = metadata.get(
    "selected_model",
    "Logistic Regression",
)

cv_auc = metadata.get(
    "cross_validation_roc_auc",
    metadata.get(
        "cv_roc_auc",
        None,
    ),
)

dataset_records = metadata.get(
    "dataset_records",
    metadata.get(
        "records",
        "N/A",
    ),
)

train_records = metadata.get(
    "train_records",
    "N/A",
)

holdout_records = metadata.get(
    "holdout_records",
    "N/A",
)

metric_cols = st.columns(4)

with metric_cols[0]:
    st.metric(
        "Selected Model",
        str(selected_model),
    )

with metric_cols[1]:
    if cv_auc is not None:
        try:
            st.metric(
                "Cross-validation ROC-AUC",
                f"{float(cv_auc):.3f}",
            )
        except Exception:
            st.metric(
                "Cross-validation ROC-AUC",
                str(cv_auc),
            )
    else:
        st.metric(
            "Cross-validation ROC-AUC",
            "N/A",
        )

with metric_cols[2]:
    st.metric(
        "Training Records",
        str(train_records),
    )

with metric_cols[3]:
    st.metric(
        "Holdout Records",
        str(holdout_records),
    )


# ============================================================
# PREDICTION
# ============================================================

st.header("1. Heart Disease Prediction")

st.markdown(
    """
    Enter the 13 model features below. The model returns a binary
    prediction and the estimated probability of Class 1.
    """
)

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
        )

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: (
                "Female (0)" if x == 0 else "Male (1)"
            ),
        )

        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[1, 2, 3, 4],
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=300,
            value=120,
        )

        chol = st.number_input(
            "Serum Cholesterol",
            min_value=50,
            max_value=700,
            value=200,
        )

    with col2:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            ),
        )

        restecg = st.selectbox(
            "Resting ECG",
            options=[0, 1, 2],
        )

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150,
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: (
                "No (0)" if x == 0 else "Yes (1)"
            ),
        )

    with col3:
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        slope = st.selectbox(
            "Slope",
            options=[1, 2, 3],
        )

        ca = st.selectbox(
            "Number of Major Vessels (ca)",
            options=[0, 1, 2, 3],
        )

        thal = st.selectbox(
            "Thal",
            options=[3, 6, 7],
        )

    submitted = st.form_submit_button(
        "Run Prediction",
        use_container_width=True,
    )


if submitted:

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
        ]
    )

    try:

        prediction = get_binary_predictions(
            model,
            input_data,
        )[0]

        probability = get_probability_for_class_1(
            model,
            input_data,
        )[0]

        st.session_state["last_prediction"] = prediction
        st.session_state["last_probability"] = probability
        st.session_state["last_input"] = input_data

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:
                st.error(
                    "Model prediction: Class 1"
                )
            else:
                st.success(
                    "Model prediction: Class 0"
                )

        with result_col2:

            st.metric(
                "Class 1 probability",
                f"{probability:.1%}",
            )

        st.warning(
            "This probability is a model output, not a clinically "
            "validated personal risk percentage."
        )

    except Exception as exc:

        st.error(
            f"Prediction could not be completed: {exc}"
        )


# ============================================================
# MODEL EVALUATION DASHBOARD
# ============================================================

st.header("2. Model Evaluation Dashboard")

if evaluation_data is None:

    st.info(
        f"`{EVALUATION_FILE}` was not found. "
        "Add the evaluation predictions file to the repository "
        "to display the holdout evaluation dashboard."
    )

else:

    required_eval_columns = [
        "y_true",
        "y_pred",
        "prob_class_1",
    ]

    missing_eval = [
        column
        for column in required_eval_columns
        if column not in evaluation_data.columns
    ]

    if missing_eval:

        st.error(
            "The evaluation CSV is missing required columns: "
            + ", ".join(missing_eval)
        )

    else:

        eval_df = evaluation_data.copy()

        y_true_eval = pd.to_numeric(
            eval_df["y_true"],
            errors="coerce",
        )

        y_pred_eval = pd.to_numeric(
            eval_df["y_pred"],
            errors="coerce",
        )

        probabilities_eval = pd.to_numeric(
            eval_df["prob_class_1"],
            errors="coerce",
        )

        valid_eval = (
            y_true_eval.notna()
            & y_pred_eval.notna()
            & probabilities_eval.notna()
        )

        y_true_eval = y_true_eval[valid_eval].astype(int).to_numpy()
        y_pred_eval = y_pred_eval[valid_eval].astype(int).to_numpy()
        probabilities_eval = (
            probabilities_eval[valid_eval].astype(float).to_numpy()
        )

        if (
            len(y_true_eval) == 0
            or len(np.unique(y_true_eval)) != 2
        ):

            st.error(
                "The evaluation data must contain both Class 0 and Class 1."
            )

        else:

            evaluation_metrics = calculate_metrics(
                y_true_eval,
                y_pred_eval,
                probabilities_eval,
            )

            metric_cols = st.columns(4)

            with metric_cols[0]:
                st.metric(
                    "Accuracy",
                    f"{evaluation_metrics['Accuracy']:.3f}",
                )

            with metric_cols[1]:
                st.metric(
                    "Precision",
                    f"{evaluation_metrics['Precision']:.3f}",
                )

            with metric_cols[2]:
                st.metric(
                    "Recall",
                    f"{evaluation_metrics['Sensitivity']:.3f}",
                )

            with metric_cols[3]:
                st.metric(
                    "F1",
                    f"{evaluation_metrics['F1 Score']:.3f}",
                )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                cm = confusion_matrix(
                    y_true_eval,
                    y_pred_eval,
                    labels=[0, 1],
                )

                cm_fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=cm,
                            x=["Predicted 0", "Predicted 1"],
                            y=["Actual 0", "Actual 1"],
                            text=cm,
                            texttemplate="%{text}",
                            colorscale="Blues",
                        )
                    ]
                )

                cm_fig.update_layout(
                    title="Confusion Matrix",
                    height=450,
                )

                st.plotly_chart(
                    cm_fig,
                    use_container_width=True,
                )

            with chart_col2:

                st.plotly_chart(
                    create_roc_figure(
                        y_true_eval,
                        probabilities_eval,
                        "Holdout ROC Curve",
                    ),
                    use_container_width=True,
                )

            summary_df = pd.DataFrame(
                {
                    "Metric": list(evaluation_metrics.keys()),
                    "Value": [
                        f"{value:.4f}"
                        for value in evaluation_metrics.values()
                    ],
                }
            )

            st.subheader("Evaluation Summary")

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Cross-validation ROC-AUC and holdout ROC-AUC are model "
                "evaluation statistics. They should not be interpreted "
                "as individual clinical risk probabilities."
            )


# ============================================================
# RELIABILITY ANALYSIS
# ============================================================

st.header("3. Reliability and Calibration Analysis")

if evaluation_data is None:

    st.info(
        "Holdout evaluation data is required for reliability analysis."
    )

else:

    try:

        eval_df = evaluation_data.copy()

        y_true_rel = pd.to_numeric(
            eval_df["y_true"],
            errors="coerce",
        )

        probabilities_rel = pd.to_numeric(
            eval_df["prob_class_1"],
            errors="coerce",
        )

        valid_rel = (
            y_true_rel.notna()
            & probabilities_rel.notna()
        )

        y_true_rel = y_true_rel[valid_rel].astype(int).to_numpy()
        probabilities_rel = (
            probabilities_rel[valid_rel].astype(float).to_numpy()
        )

        y_pred_rel = (
            probabilities_rel >= 0.5
        ).astype(int)

        reliability_metrics = calculate_metrics(
            y_true_rel,
            y_pred_rel,
            probabilities_rel,
        )

        rcols = st.columns(4)

        with rcols[0]:
            st.metric(
                "Balanced Accuracy",
                f"{reliability_metrics['Balanced Accuracy']:.3f}",
            )

        with rcols[1]:
            st.metric(
                "Sensitivity",
                f"{reliability_metrics['Sensitivity']:.3f}",
            )

        with rcols[2]:
            st.metric(
                "Specificity",
                f"{reliability_metrics['Specificity']:.3f}",
            )

        with rcols[3]:
            st.metric(
                "Brier Score",
                f"{reliability_metrics['Brier Score']:.3f}",
            )

        st.plotly_chart(
            create_calibration_figure(
                y_true_rel,
                probabilities_rel,
                "Holdout Calibration Curve",
            ),
            use_container_width=True,
        )

        report_dict = classification_report(
            y_true_rel,
            y_pred_rel,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report_dict).transpose()

        st.subheader("Classification Report")

        st.dataframe(
            report_df.round(4),
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="warning-note">
            <b>Calibration limitation:</b>
            A Brier score alone does not establish that predicted
            probabilities are clinically calibrated. Calibration should
            be assessed using the reliability curve and, ideally,
            additional calibration statistics on an appropriately
            representative validation cohort.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:

        st.error(
            f"Reliability analysis could not be completed: {exc}"
        )


# ============================================================
# PDF AND EVALUATION DOWNLOADS
# ============================================================

st.header("4. Evaluation Report and Downloads")

if evaluation_data is not None:

    try:

        y_true_report = pd.to_numeric(
            evaluation_data["y_true"],
            errors="coerce",
        )

        y_pred_report = pd.to_numeric(
            evaluation_data["y_pred"],
            errors="coerce",
        )

        probabilities_report = pd.to_numeric(
            evaluation_data["prob_class_1"],
            errors="coerce",
        )

        valid_report = (
            y_true_report.notna()
            & y_pred_report.notna()
            & probabilities_report.notna()
        )

        y_true_report = y_true_report[valid_report].astype(int).to_numpy()
        y_pred_report = y_pred_report[valid_report].astype(int).to_numpy()
        probabilities_report = (
            probabilities_report[valid_report].astype(float).to_numpy()
        )

        report_metrics = calculate_metrics(
            y_true_report,
            y_pred_report,
            probabilities_report,
        )

        report_rows = []

        for metric_name, value in report_metrics.items():
            report_rows.append(
                {
                    "Metric": metric_name,
                    "Value": value,
                }
            )

        report_download_df = pd.DataFrame(report_rows)

        st.download_button(
            label="Download Evaluation Metrics CSV",
            data=report_download_df.to_csv(index=False).encode("utf-8"),
            file_name="afrihealth_evaluation_metrics.csv",
            mime="text/csv",
        )

        if REPORTLAB_AVAILABLE:

            pdf_bytes = create_pdf_report(
                source="AfriHealth AI holdout evaluation dataset",
                n_records=len(y_true_report),
                class_0=int(np.sum(y_true_report == 0)),
                class_1=int(np.sum(y_true_report == 1)),
                metrics_df=pd.DataFrame(
                    [
                        {
                            "Metric": key,
                            "Estimate": value,
                            "95% CI Lower": np.nan,
                            "95% CI Upper": np.nan,
                        }
                        for key, value in report_metrics.items()
                    ]
                ),
                calibration_text=(
                    "The holdout calibration curve is provided in the "
                    "interactive application. This PDF does not establish "
                    "clinical calibration."
                ),
                constant_features=[],
            )

            if pdf_bytes:

                st.download_button(
                    label="Download Evaluation PDF Report",
                    data=pdf_bytes,
                    file_name="afrihealth_evaluation_report.pdf",
                    mime="application/pdf",
                )

        else:

            st.info(
                "ReportLab is not available. Install the required "
                "dependency from requirements.txt to enable PDF reports."
            )

    except Exception as exc:

        st.error(
            f"Report generation failed: {exc}"
        )


# ============================================================
# EXPLAINABILITY
# ============================================================

st.header("5. Explainability")

st.subheader("Global Logistic Regression Coefficients")

coefficient_df = get_logistic_coefficients(model)

if coefficient_df is None:

    st.info(
        "Coefficient-based explainability is available when the "
        "loaded model contains a compatible Logistic Regression estimator."
    )

else:

    display_coefficients = coefficient_df[
        [
            "Feature",
            "Coefficient",
            "Absolute Effect",
        ]
    ].copy()

    display_coefficients["Coefficient"] = (
        display_coefficients["Coefficient"].round(4)
    )

    display_coefficients["Absolute Effect"] = (
        display_coefficients["Absolute Effect"].round(4)
    )

    st.dataframe(
        display_coefficients,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Positive coefficients increase the model's linear tendency "
        "toward Class 1. Negative coefficients decrease it. These are "
        "model coefficients, not clinical causal effects."
    )


if "last_input" in st.session_state:

    st.subheader("Local Prediction Explanation")

    local_result = get_local_contributions(
        model,
        st.session_state["last_input"],
    )

    if local_result is None:

        st.info(
            "A local contribution explanation could not be generated "
            "for the loaded model."
        )

    else:

        local_display = local_result[
            [
                "Feature",
                "Contribution",
                "Absolute Contribution",
            ]
        ].copy()

        local_display["Contribution"] = (
            local_display["Contribution"].round(4)
        )

        local_display["Absolute Contribution"] = (
            local_display["Absolute Contribution"].round(4)
        )

        st.dataframe(
            local_display,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Local contributions show how transformed input features "
            "contribute to the logistic model's Class 1 score. They "
            "should not be interpreted as causal clinical explanations."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("6. Model Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.subheader("Model")

    st.write(
        f"**Selected model:** {selected_model}"
    )

    if metadata:

        sklearn_version = metadata.get(
            "sklearn_version",
            metadata.get(
                "scikit_learn_version",
                "Not supplied",
            ),
        )

        st.write(
            f"**scikit-learn version:** {sklearn_version}"
        )

        if cv_auc is not None:
            try:
                st.write(
                    f"**Cross-validation ROC-AUC:** {float(cv_auc):.4f}"
                )
            except Exception:
                st.write(
                    f"**Cross-validation ROC-AUC:** {cv_auc}"
                )

        st.write(
            f"**Dataset records:** {dataset_records}"
        )

with info_col2:

    st.subheader("Feature Schema")

    st.dataframe(
        pd.DataFrame(
            {
                "Feature": DEFAULT_FEATURES,
                "Expected range": [
                    f"{EXPECTED_FEATURE_RANGES[f][0]}–"
                    f"{EXPECTED_FEATURE_RANGES[f][1]}"
                    for f in DEFAULT_FEATURES
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EXTERNAL VALIDATION
# ============================================================

st.header("7. Build 06A: External Validation Workspace")

st.markdown(
    """
    This workspace evaluates the frozen AfriHealth AI model on an
    external dataset without retraining the model.

    The external dataset must contain the same 13 features expected
    by the model plus a binary `target` column.
    """
)

st.markdown(
    """
    <div class="research-note">
    <b>External validation principle:</b>
    The model is frozen during this evaluation. No model retraining,
    parameter fitting or threshold optimization is performed here.
    </div>
    """,
    unsafe_allow_html=True,
)

source_default = (
    "UCI Heart Disease Dataset, Hungarian cohort"
)

external_source = st.text_input(
    "External dataset source / provenance",
    value=source_default,
    help=(
        "Enter the repository, cohort, institution or study source. "
        "Do not leave this as 'Not supplied' when preparing a research report."
    ),
)

uploaded_file = st.file_uploader(
    "Upload external validation CSV",
    type=["csv"],
    key="external_validation_uploader",
)

if uploaded_file is not None:

    try:

        external_df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:

        st.error(
            f"Could not read the uploaded CSV: {exc}"
        )

        external_df = None

    if external_df is not None:

        required_external_columns = DEFAULT_FEATURES + ["target"]

        missing_columns = [
            column
            for column in required_external_columns
            if column not in external_df.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )

        else:

            # Keep only required columns for evaluation.
            external_working = external_df[
                required_external_columns
            ].copy()

            # Convert every required column to numeric.
            for column in required_external_columns:

                external_working[column] = pd.to_numeric(
                    external_working[column],
                    errors="coerce",
                )

            missing_after_conversion = (
                external_working.isna().sum()
            )

            columns_with_missing = (
                missing_after_conversion[
                    missing_after_conversion > 0
                ]
                .index
                .tolist()
            )

            if columns_with_missing:

                st.error(
                    "The uploaded dataset contains missing or non-numeric "
                    "values after conversion in: "
                    + ", ".join(columns_with_missing)
                )

            else:

                # Validate target.
                target_values = set(
                    external_working["target"]
                    .astype(int)
                    .unique()
                    .tolist()
                )

                if not target_values.issubset({0, 1}):

                    st.error(
                        "The `target` column must contain only 0 and 1."
                    )

                elif len(target_values) != 2:

                    st.error(
                        "The external validation dataset must contain "
                        "both Class 0 and Class 1."
                    )

                else:

                    # Validate feature ranges.
                    range_warnings = []

                    for feature_name in DEFAULT_FEATURES:

                        minimum, maximum = (
                            EXPECTED_FEATURE_RANGES[
                                feature_name
                            ]
                        )

                        values = external_working[
                            feature_name
                        ]

                        outside = (
                            (values < minimum)
                            | (values > maximum)
                        )

                        if outside.any():

                            range_warnings.append(
                                f"{feature_name}: "
                                f"{int(outside.sum())} value(s) "
                                f"outside expected range "
                                f"{minimum}–{maximum}"
                            )

                    if range_warnings:

                        st.warning(
                            "Some external feature values are outside "
                            "the expected model input ranges:\n\n"
                            + "\n".join(
                                [
                                    f"- {warning}"
                                    for warning in range_warnings
                                ]
                            )
                        )

                    # Identify constant features.
                    constant_features = []

                    for feature_name in DEFAULT_FEATURES:

                        if (
                            external_working[
                                feature_name
                            ].nunique()
                            <= 1
                        ):

                            constant_features.append(
                                feature_name
                            )

                    if constant_features:

                        st.warning(
                            "Constant feature warning: "
                            + ", ".join(constant_features)
                            + " has no variation in this external "
                            "dataset. This may limit the usefulness "
                            "of external generalisation assessment."
                        )

                    # Display dataset information.
                    st.subheader(
                        "External Dataset Summary"
                    )

                    external_n = len(
                        external_working
                    )

                    external_class_0 = int(
                        (
                            external_working["target"]
                            == 0
                        ).sum()
                    )

                    external_class_1 = int(
                        (
                            external_working["target"]
                            == 1
                        ).sum()
                    )

                    summary_cols = st.columns(4)

                    with summary_cols[0]:
                        st.metric(
                            "Records evaluated",
                            external_n,
                        )

                    with summary_cols[1]:
                        st.metric(
                            "Class 0",
                            external_class_0,
                        )

                    with summary_cols[2]:
                        st.metric(
                            "Class 1",
                            external_class_1,
                        )

                    with summary_cols[3]:
                        st.metric(
                            "Class 1 proportion",
                            f"{external_class_1 / external_n:.1%}",
                        )

                    st.write(
                        f"**Dataset source:** "
                        f"{external_source.strip() or 'Not supplied'}"
                    )

                    st.dataframe(
                        external_working.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ------------------------------------------------
                    # FROZEN MODEL EXTERNAL PREDICTION
                    # ------------------------------------------------

                    X_external = external_working[
                        DEFAULT_FEATURES
                    ].copy()

                    y_external = (
                        external_working["target"]
                        .astype(int)
                        .to_numpy()
                    )

                    try:

                        external_predictions = (
                            get_binary_predictions(
                                model,
                                X_external,
                            )
                        )

                        external_probabilities = (
                            get_probability_for_class_1(
                                model,
                                X_external,
                            )
                        )

                        # Validate probability values.
                        if not np.all(
                            np.isfinite(
                                external_probabilities
                            )
                        ):

                            raise ValueError(
                                "The model returned non-finite "
                                "probabilities."
                            )

                        if np.any(
                            (external_probabilities < 0)
                            | (external_probabilities > 1)
                        ):

                            raise ValueError(
                                "The model returned probabilities "
                                "outside the 0–1 range."
                            )

                        # Validate prediction count.
                        if len(external_predictions) != external_n:

                            raise ValueError(
                                "The number of model predictions does "
                                "not match the number of external records."
                            )

                    except Exception as exc:

                        st.error(
                            "External prediction failed: "
                            f"{exc}"
                        )

                        external_predictions = None
                        external_probabilities = None

                    # ------------------------------------------------
                    # EXTERNAL METRICS
                    # ------------------------------------------------

                    if (
                        external_predictions is not None
                        and external_probabilities is not None
                    ):

                        external_metrics = calculate_metrics(
                            y_external,
                            external_predictions,
                            external_probabilities,
                        )

                        st.subheader(
                            "External Validation Performance"
                        )

                        metric_cols = st.columns(4)

                        with metric_cols[0]:
                            st.metric(
                                "Accuracy",
                                f"{external_metrics['Accuracy']:.3f}",
                            )

                        with metric_cols[1]:
                            st.metric(
                                "Balanced Accuracy",
                                f"{external_metrics['Balanced Accuracy']:.3f}",
                            )

                        with metric_cols[2]:
                            st.metric(
                                "Sensitivity",
                                f"{external_metrics['Sensitivity']:.3f}",
                            )

                        with metric_cols[3]:
                            st.metric(
                                "Specificity",
                                f"{external_metrics['Specificity']:.3f}",
                            )

                        metric_cols_2 = st.columns(4)

                        with metric_cols_2[0]:
                            st.metric(
                                "Precision",
                                f"{external_metrics['Precision']:.3f}",
                            )

                        with metric_cols_2[1]:
                            st.metric(
                                "F1 Score",
                                f"{external_metrics['F1 Score']:.3f}",
                            )

                        with metric_cols_2[2]:
                            st.metric(
                                "ROC-AUC",
                                f"{external_metrics['ROC-AUC']:.3f}",
                            )

                        with metric_cols_2[3]:
                            st.metric(
                                "Brier Score",
                                f"{external_metrics['Brier Score']:.3f}",
                            )

                        # ------------------------------------------------
                        # MATHEMATICAL CONSISTENCY CHECK
                        # ------------------------------------------------

                        calculated_balanced_accuracy = (
                            external_metrics["Sensitivity"]
                            + external_metrics["Specificity"]
                        ) / 2

                        balanced_accuracy_difference = abs(
                            calculated_balanced_accuracy
                            - external_metrics["Balanced Accuracy"]
                        )

                        if balanced_accuracy_difference > 1e-8:

                            st.error(
                                "Metric consistency check failed. "
                                "Balanced accuracy does not equal the "
                                "mean of sensitivity and specificity."
                            )

                        else:

                            st.success(
                                "Metric consistency check passed. "
                                "Sensitivity, specificity and balanced "
                                "accuracy were calculated from the same "
                                "predictions."
                            )

                        # ------------------------------------------------
                        # CONFUSION MATRIX
                        # ------------------------------------------------

                        st.subheader(
                            "External Confusion Matrix"
                        )

                        external_cm = confusion_matrix(
                            y_external,
                            external_predictions,
                            labels=[0, 1],
                        )

                        cm_fig = go.Figure(
                            data=[
                                go.Heatmap(
                                    z=external_cm,
                                    x=[
                                        "Predicted 0",
                                        "Predicted 1",
                                    ],
                                    y=[
                                        "Actual 0",
                                        "Actual 1",
                                    ],
                                    text=external_cm,
                                    texttemplate="%{text}",
                                    colorscale="Blues",
                                )
                            ]
                        )

                        cm_fig.update_layout(
                            title="External Validation Confusion Matrix",
                            height=500,
                        )

                        st.plotly_chart(
                            cm_fig,
                            use_container_width=True,
                        )

                        # ------------------------------------------------
                        # CLASSIFICATION REPORT
                        # ------------------------------------------------

                        st.subheader(
                            "External Classification Report"
                        )

                        external_report = classification_report(
                            y_external,
                            external_predictions,
                            output_dict=True,
                            zero_division=0,
                        )

                        external_report_df = (
                            pd.DataFrame(
                                external_report
                            ).transpose()
                        )

                        st.dataframe(
                            external_report_df.round(4),
                            use_container_width=True,
                        )

                        # ------------------------------------------------
                        # BOOTSTRAP CONFIDENCE INTERVALS
                        # ------------------------------------------------

                        st.subheader(
                            "95% Bootstrap Confidence Intervals"
                        )

                        st.caption(
                            "Intervals are generated using 1,000 "
                            "class-stratified bootstrap resamples. "
                            "They quantify sampling uncertainty within "
                            "this external evaluation cohort."
                        )

                        with st.spinner(
                            "Calculating 1,000 class-stratified bootstrap resamples..."
                        ):

                            external_metrics_ci = (
                                create_bootstrap_results(
                                    y_external,
                                    external_predictions,
                                    external_probabilities,
                                    n_bootstrap=1000,
                                )
                            )

                        ci_display = (
                            external_metrics_ci.copy()
                        )

                        ci_display[
                            "Estimate"
                        ] = ci_display[
                            "Estimate"
                        ].map(
                            lambda value: (
                                f"{value:.3f}"
                                if pd.notna(value)
                                else "N/A"
                            )
                        )

                        ci_display[
                            "95% CI"
                        ] = ci_display.apply(
                            lambda row: format_ci(
                                row["95% CI Lower"],
                                row["95% CI Upper"],
                            ),
                            axis=1,
                        )

                        ci_display = ci_display[
                            [
                                "Metric",
                                "Estimate",
                                "95% CI",
                            ]
                        ]

                        st.dataframe(
                            ci_display,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ------------------------------------------------
                        # ROC CURVE
                        # ------------------------------------------------

                        st.subheader(
                            "External ROC Curve"
                        )

                        st.plotly_chart(
                            create_roc_figure(
                                y_external,
                                external_probabilities,
                                "External Validation ROC Curve",
                            ),
                            use_container_width=True,
                        )

                        # ------------------------------------------------
                        # EXTERNAL CALIBRATION
                        # ------------------------------------------------

                        st.subheader(
                            "External Calibration Curve"
                        )

                        st.plotly_chart(
                            create_calibration_figure(
                                y_external,
                                external_probabilities,
                                "External Validation Calibration Curve",
                            ),
                            use_container_width=True,
                        )

                        calibration_interpretation = (
                            interpret_calibration(
                                y_external,
                                external_probabilities,
                            )
                        )

                        st.info(
                            calibration_interpretation
                        )

                        st.warning(
                            "The Brier score and calibration curve are "
                            "descriptive evidence of probabilistic "
                            "performance. They do not establish that "
                            "AfriHealth AI provides clinically calibrated "
                            "individual risk."
                        )

                        # ------------------------------------------------
                        # EXTERNAL VALIDATION INTERPRETATION
                        # ------------------------------------------------

                        st.subheader(
                            "Research Interpretation"
                        )

                        auc_value = external_metrics[
                            "ROC-AUC"
                        ]

                        accuracy_value = external_metrics[
                            "Accuracy"
                        ]

                        sensitivity_value = external_metrics[
                            "Sensitivity"
                        ]

                        specificity_value = external_metrics[
                            "Specificity"
                        ]

                        brier_value = external_metrics[
                            "Brier Score"
                        ]

                        st.markdown(
                            f"""
                            **AfriHealth AI demonstrated an ROC-AUC of
                            {auc_value:.3f} on the evaluated external
                            cohort of {external_n} records.**

                            The model achieved an accuracy of
                            {accuracy_value:.1%}, sensitivity of
                            {sensitivity_value:.1%}, and specificity of
                            {specificity_value:.1%} at the model's
                            classification threshold.

                            The Brier score was {brier_value:.3f}.
                            Probabilistic performance should be interpreted
                            together with the calibration curve rather than
                            from the Brier score alone.

                            These results indicate promising performance
                            on this particular external cohort. They do
                            not establish clinical validity, clinical
                            safety, or generalisability to Nigerian
                            populations.
                            """
                        )

                        # ------------------------------------------------
                        # DATA QUALITY / GENERALISATION WARNINGS
                        # ------------------------------------------------

                        st.subheader(
                            "External Validation Limitations"
                        )

                        limitation_items = [
                            "The model is being evaluated on a public external cohort and was not retrained during this analysis.",
                            "The current model has not been specifically trained or clinically validated on a Nigerian population.",
                            "Differences in patient demographics, measurement practices, disease prevalence and feature distributions may affect generalisation.",
                            "The external dataset contains constant features where applicable. Such features provide no variation for assessing external robustness.",
                            "Confidence intervals describe sampling uncertainty in this cohort and do not establish population-level clinical validity.",
                            "The selected classification threshold may not be optimal for a clinical screening or diagnostic use case.",
                            "The model's predicted probabilities should not be interpreted as clinically calibrated individual risk.",
                            "Prospective validation and appropriate clinical governance would be required before clinical deployment.",
                        ]

                        for item in limitation_items:
                            st.markdown(
                                f"- {item}"
                            )

                        # ------------------------------------------------
                        # DOWNLOADABLE METRICS
                        # ------------------------------------------------

                        st.subheader(
                            "Download External Validation Results"
                        )

                        metrics_download_df = (
                            external_metrics_ci.copy()
                        )

                        metrics_csv = (
                            make_metrics_download(
                                metrics_download_df
                            )
                        )

                        st.download_button(
                            label="Download External Metrics CSV",
                            data=metrics_csv,
                            file_name=(
                                "afrihealth_external_validation_metrics.csv"
                            ),
                            mime="text/csv",
                        )

                        # ------------------------------------------------
                        # DOWNLOAD PREDICTIONS
                        # ------------------------------------------------

                        predictions_csv = (
                            make_external_predictions_download(
                                external_working,
                                external_predictions,
                                external_probabilities,
                            )
                        )

                        st.download_button(
                            label="Download External Predictions CSV",
                            data=predictions_csv,
                            file_name=(
                                "afrihealth_external_validation_predictions.csv"
                            ),
                            mime="text/csv",
                        )

                        # ------------------------------------------------
                        # PDF REPORT
                        # ------------------------------------------------

                        if REPORTLAB_AVAILABLE:

                            pdf_metrics = (
                                external_metrics_ci.copy()
                            )

                            pdf_bytes = create_pdf_report(
                                source=(
                                    external_source.strip()
                                    or "Not supplied"
                                ),
                                n_records=external_n,
                                class_0=external_class_0,
                                class_1=external_class_1,
                                metrics_df=pdf_metrics,
                                calibration_text=(
                                    calibration_interpretation
                                ),
                                constant_features=(
                                    constant_features
                                ),
                            )

                            if pdf_bytes:

                                st.download_button(
                                    label=(
                                        "Download External Validation "
                                        "PDF Report"
                                    ),
                                    data=pdf_bytes,
                                    file_name=(
                                        "afrihealth_external_validation_report.pdf"
                                    ),
                                    mime="application/pdf",
                                )

                        else:

                            st.info(
                                "PDF reporting requires ReportLab. "
                                "Ensure `reportlab>=4.0,<5` is present "
                                "in requirements.txt."
                            )


# ============================================================
# RESEARCH LIMITATIONS
# ============================================================

st.header("8. Research Limitations")

limitations = [
    "AfriHealth AI is a research prototype and not a clinically approved medical device.",
    "The current model is based on the UCI Heart Disease dataset.",
    "The model has not been specifically trained or validated on Nigerian or broader African populations.",
    "External validation on a public cohort does not establish clinical validity.",
    "Model performance may change because of dataset shift, demographic differences, clinical practice differences and measurement differences.",
    "Predicted probabilities are not automatically calibrated clinical risk estimates.",
    "The model should not be used to diagnose disease or determine treatment.",
    "Future validation should use appropriately governed local clinical data.",
    "Any future clinical study should have a predefined statistical analysis plan and appropriate ethical and data-protection approvals.",
]

for item in limitations:
    st.markdown(
        f"- {item}"
    )


# ============================================================
# RESEARCH ROADMAP
# ============================================================

st.header("9. Recommended Research Roadmap")

roadmap = [
    (
        "Stage 1",
        "Public external validation",
        "Completed as an initial generalisation assessment."
    ),
    (
        "Stage 2",
        "Metric and calibration audit",
        "Confirm reproducibility, confidence intervals and calibration."
    ),
    (
        "Stage 3",
        "Nigerian retrospective validation",
        "Evaluate the frozen model on appropriately governed Nigerian clinical data."
    ),
    (
        "Stage 4",
        "Prospective validation",
        "Evaluate performance prospectively under a predefined study protocol."
    ),
    (
        "Stage 5",
        "Clinical utility assessment",
        "Assess whether the model improves decision-making compared with existing practice."
    ),
]

roadmap_df = pd.DataFrame(
    roadmap,
    columns=[
        "Stage",
        "Research Phase",
        "Objective",
    ],
)

st.dataframe(
    roadmap_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PHASE 6B: DATA QUALITY & DISTRIBUTION AUDIT
# ============================================================

with st.expander(
    "Phase 6B: Data Quality & Distribution Audit",
    expanded=False
):
    import numpy as np
    import pandas as pd
    import plotly.express as px

    st.subheader("Research Data Audit")

    st.write(
        "Compare development, holdout, and external datasets "
        "without retraining or modifying the frozen model."
    )

    FEATURES_AUDIT = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ]

    TARGET_AUDIT = "target"

    st.warning(
        "Upload datasets only when you have verified their "
        "source, cohort, feature definitions, and target coding. "
        "This audit does not establish clinical validity."
    )

    def audit_dataset(df, dataset_name):
        """Create row-level quality summaries for a dataset."""
        missing_features = [
            col for col in FEATURES_AUDIT if col not in df.columns
        ]

        if missing_features:
            return None, (
                f"{dataset_name}: missing required columns: "
                + ", ".join(missing_features)
            )

        work = df[FEATURES_AUDIT].copy()

        for col in FEATURES_AUDIT:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        rows = []

        for col in FEATURES_AUDIT:
            values = work[col]
            observed = values.dropna()

            rows.append({
                "Dataset": dataset_name,
                "Feature": col,
                "Records": len(work),
                "Missing": int(values.isna().sum()),
                "Missing (%)": round(
                    values.isna().mean() * 100, 2
                ),
                "Unique values": int(observed.nunique()),
                "Constant feature": (
                    observed.nunique() <= 1
                    if len(observed) > 0 else True
                ),
                "Minimum": (
                    observed.min() if len(observed) else np.nan
                ),
                "Median": (
                    observed.median() if len(observed) else np.nan
                ),
                "Maximum": (
                    observed.max() if len(observed) else np.nan
                ),
                "Mean": (
                    observed.mean() if len(observed) else np.nan
                ),
                "Standard deviation": (
                    observed.std() if len(observed) > 1 else np.nan
                ),
            })

        summary = pd.DataFrame(rows)

        dataset_summary = {
            "Dataset": dataset_name,
            "Records": len(df),
            "Duplicate rows": int(df.duplicated().sum()),
            "Missing target": (
                int(df[TARGET_AUDIT].isna().sum())
                if TARGET_AUDIT in df.columns else np.nan
            ),
            "Target classes": (
                int(df[TARGET_AUDIT].nunique(dropna=True))
                if TARGET_AUDIT in df.columns else np.nan
            ),
        }

        if TARGET_AUDIT in df.columns:
            target = pd.to_numeric(
                df[TARGET_AUDIT], errors="coerce"
            )
            valid_target = target.dropna()
            dataset_summary["Invalid target values"] = int(
                (~valid_target.isin([0, 1])).sum()
            )
            dataset_summary["Class 0 (%)"] = round(
                (target == 0).mean() * 100, 2
            )
            dataset_summary["Class 1 (%)"] = round(
                (target == 1).mean() * 100, 2
            )
        else:
            dataset_summary["Invalid target values"] = np.nan
            dataset_summary["Class 0 (%)"] = np.nan
            dataset_summary["Class 1 (%)"] = np.nan

        return (summary, dataset_summary), None

    st.markdown("### 1. Upload comparison datasets")

    st.caption(
        "Upload CSV files containing the 13 model features and, "
        "where available, a binary target column coded 0 or 1."
    )

    development_file = st.file_uploader(
        "Development or training dataset (CSV)",
        type=["csv"],
        key="audit_development"
    )

    holdout_file = st.file_uploader(
        "Internal holdout dataset (CSV)",
        type=["csv"],
        key="audit_holdout"
    )

    external_file = st.file_uploader(
        "External validation dataset (CSV)",
        type=["csv"],
        key="audit_external"
    )

    uploaded_sets = []

    for label, uploaded in [
        ("Development", development_file),
        ("Internal holdout", holdout_file),
        ("External validation", external_file),
    ]:
        if uploaded is not None:
            try:
                uploaded.seek(0)
                frame = pd.read_csv(uploaded)
                uploaded_sets.append((label, frame))
            except Exception as exc:
                st.error(f"Could not read {label} CSV: {exc}")

    if uploaded_sets:
        all_feature_summaries = []
        all_dataset_summaries = []
        valid_frames = {}

        for label, frame in uploaded_sets:
            result, error = audit_dataset(frame, label)

            if error:
                st.error(error)
                continue

            feature_summary, dataset_summary = result
            all_feature_summaries.append(feature_summary)
            all_dataset_summaries.append(dataset_summary)

            clean = frame.copy()
            for col in FEATURES_AUDIT:
                clean[col] = pd.to_numeric(
                    clean[col], errors="coerce"
                )
            valid_frames[label] = clean

        if all_feature_summaries:
            feature_audit = pd.concat(
                all_feature_summaries, ignore_index=True
            )
            dataset_audit = pd.DataFrame(all_dataset_summaries)

            st.markdown("### 2. Dataset overview")
            st.dataframe(dataset_audit, use_container_width=True)

            st.markdown("### 3. Feature quality")
            st.dataframe(feature_audit, use_container_width=True)

            st.download_button(
                "Download feature quality audit",
                feature_audit.to_csv(index=False).encode("utf-8"),
                file_name="afrihealth_phase6b_feature_audit.csv",
                mime="text/csv",
                key="download_feature_audit"
            )

            st.download_button(
                "Download dataset summary",
                dataset_audit.to_csv(index=False).encode("utf-8"),
                file_name="afrihealth_phase6b_dataset_summary.csv",
                mime="text/csv",
                key="download_dataset_audit"
            )

            st.markdown("### 4. Compare feature distributions")

            selected_feature = st.selectbox(
                "Select a feature",
                FEATURES_AUDIT,
                key="audit_feature_selection"
            )

            plot_frames = []

            for label, frame in valid_frames.items():
                values = frame[selected_feature].dropna()

                if not values.empty:
                    plot_frames.append(pd.DataFrame({
                        "Dataset": label,
                        "Value": values
                    }))

            if plot_frames:
                plot_data = pd.concat(
                    plot_frames, ignore_index=True
                )

                fig = px.box(
                    plot_data,
                    x="Dataset",
                    y="Value",
                    color="Dataset",
                    points="outliers",
                    title=(
                        f"{selected_feature}: "
                        "dataset distribution comparison"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 5. Standardised mean differences")

            reference_name = st.selectbox(
                "Reference dataset",
                list(valid_frames.keys()),
                key="audit_reference"
            )

            reference = valid_frames[reference_name]
            smd_rows = []

            for label, frame in valid_frames.items():
                if label == reference_name:
                    continue

                for feature in FEATURES_AUDIT:
                    a = pd.to_numeric(
                        reference[feature], errors="coerce"
                    ).dropna()
                    b = pd.to_numeric(
                        frame[feature], errors="coerce"
                    ).dropna()

                    if len(a) < 2 or len(b) < 2:
                        smd = np.nan
                    else:
                        pooled_sd = np.sqrt(
                            (a.var(ddof=1) + b.var(ddof=1)) / 2
                        )

                        smd = (
                            (b.mean() - a.mean()) / pooled_sd
                            if pooled_sd > 0 else np.nan
                        )

                    smd_rows.append({
                        "Reference": reference_name,
                        "Comparison": label,
                        "Feature": feature,
                        "SMD": smd,
                        "Absolute SMD": (
                            abs(smd) if pd.notna(smd) else np.nan
                        ),
                    })

            smd_table = pd.DataFrame(smd_rows)

            if not smd_table.empty:
                smd_table["Potential shift"] = (
                    smd_table["Absolute SMD"] >= 0.20
                )

                st.dataframe(
                    smd_table.sort_values(
                        "Absolute SMD",
                        ascending=False,
                        na_position="last"
                    ),
                    use_container_width=True
                )

                st.caption(
                    "Absolute SMD values around 0.20, 0.50, and "
                    "0.80 are sometimes used as descriptive "
                    "small, moderate, and large difference "
                    "benchmarks. They are not universal pass/fail "
                    "criteria, especially for binary or categorical "
                    "features."
                )

                st.download_button(
                    "Download distribution comparison",
                    smd_table.to_csv(index=False).encode("utf-8"),
                    file_name="afrihealth_phase6b_distribution_comparison.csv",
                    mime="text/csv",
                    key="download_smd_audit"
                )

            st.markdown("### 6. Research interpretation")

            st.write(
                "- Missing values may indicate data quality "
                "problems or differences in collection procedures."
            )
            st.write(
                "- Constant features provide no within-dataset "
                "variation for assessing their distribution."
            )
            st.write(
                "- Large feature differences may reflect population "
                "differences, measurement differences, coding "
                "differences, or data processing issues."
            )
            st.write(
                "- These comparisons do not prove that distribution "
                "shift caused a change in model performance."
            )
            st.write(
                "- Do not pool cohorts or change the frozen model "
                "solely to improve external metrics."
            )

            if "External validation" in valid_frames:
                external = valid_frames["External validation"]

                constant_features = [
                    feature for feature in FEATURES_AUDIT
                    if external[feature].nunique(dropna=True) <= 1
                ]

                if constant_features:
                    st.warning(
                        "External dataset constant features: "
                        + ", ".join(constant_features)
                        + ". Check source documentation and coding "
                        "before drawing conclusions."
                    )

                if "ca" in constant_features:
                    st.error(
                        "The external dataset has no variation in "
                        "'ca'. This limits the assessment of this "
                        "predictor in the external cohort. Verify "
                        "the original source and parsing."
                    )

    else:
        st.info(
            "Upload at least one dataset to begin the audit. "
            "For a three-way comparison, upload the development, "
            "internal holdout, and external validation CSV files."
        )


# ============================================================
# PHASE 6C: EXPLORATORY CLASSIFICATION THRESHOLD ANALYSIS
# ============================================================

with st.expander(
    "Phase 6C: Threshold Analysis",
    expanded=False
):
    st.subheader("Explore classification thresholds")

    st.write(
        "This analysis examines how different probability "
        "thresholds affect classification performance. It uses "
        "the existing external validation predictions and does "
        "not retrain the model."
    )

    st.warning(
        "Research use only. Threshold results on the external "
        "cohort are exploratory. Do not select a threshold using "
        "this cohort and then describe its performance as "
        "independent validation. This tool is not for clinical "
        "decision-making."
    )

    # Import metrics locally to avoid relying on existing imports.
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    # These variables are created by the existing external
    # validation section when a valid CSV has been evaluated.
    if (
        "y_external" in locals()
        and "external_probabilities" in locals()
    ):
        threshold_y = np.asarray(y_external).reshape(-1).astype(int)
        threshold_prob = np.asarray(
            external_probabilities,
            dtype=float
        ).reshape(-1)

        valid_data = (
            len(threshold_y) == len(threshold_prob)
            and len(threshold_y) > 0
            and set(np.unique(threshold_y)).issubset({0, 1})
            and len(np.unique(threshold_y)) == 2
            and np.isfinite(threshold_prob).all()
            and ((threshold_prob >= 0) & (threshold_prob <= 1)).all()
        )

        if not valid_data:
            st.error(
                "The external labels or probabilities are invalid. "
                "Return to External Validation and check the dataset."
            )
        else:
            thresholds = np.round(
                np.arange(0.10, 0.901, 0.05),
                2
            )

            results = []

            for threshold in thresholds:
                predicted = (
                    threshold_prob >= threshold
                ).astype(int)

                tn, fp, fn, tp = confusion_matrix(
                    threshold_y,
                    predicted,
                    labels=[0, 1]
                ).ravel()

                sensitivity = recall_score(
                    threshold_y,
                    predicted,
                    zero_division=0
                )

                specificity = (
                    tn / (tn + fp)
                    if (tn + fp) > 0
                    else 0.0
                )

                results.append({
                    "Threshold": threshold,
                    "Accuracy": accuracy_score(
                        threshold_y, predicted
                    ),
                    "Balanced accuracy": balanced_accuracy_score(
                        threshold_y, predicted
                    ),
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "Precision": precision_score(
                        threshold_y,
                        predicted,
                        zero_division=0
                    ),
                    "F1 score": f1_score(
                        threshold_y,
                        predicted,
                        zero_division=0
                    ),
                    "True negatives": int(tn),
                    "False positives": int(fp),
                    "False negatives": int(fn),
                    "True positives": int(tp)
                })

            threshold_df = pd.DataFrame(results)

            st.markdown("### Threshold performance")

            st.caption(
                "The default classification threshold is 0.50. "
                "Lower thresholds generally classify more records "
                "as positive; higher thresholds generally classify "
                "fewer records as positive."
            )

            display_df = threshold_df.copy()

            metric_columns = [
                "Accuracy",
                "Balanced accuracy",
                "Sensitivity",
                "Specificity",
                "Precision",
                "F1 score"
            ]

            display_df[metric_columns] = (
                display_df[metric_columns].round(3)
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Compare performance across thresholds")

            chart_df = threshold_df.melt(
                id_vars=["Threshold"],
                value_vars=[
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "F1 score",
                    "Balanced accuracy"
                ],
                var_name="Metric",
                value_name="Score"
            )

            fig_threshold = px.line(
                chart_df,
                x="Threshold",
                y="Score",
                color="Metric",
                markers=True,
                title="Performance metrics by classification threshold"
            )

            fig_threshold.update_layout(
                yaxis=dict(
                    title="Score",
                    range=[0, 1]
                ),
                xaxis=dict(
                    title="Probability threshold",
                    tickformat=".2f"
                ),
                legend_title_text="Metric"
            )

            st.plotly_chart(
                fig_threshold,
                use_container_width=True
            )

            # Highlight the existing 0.50 threshold for comparison.
            baseline = threshold_df.loc[
                np.isclose(threshold_df["Threshold"], 0.50)
            ]

            if not baseline.empty:
                baseline_row = baseline.iloc[0]

                st.markdown(
                    "### Baseline: threshold 0.50"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Sensitivity",
                    f"{baseline_row['Sensitivity']:.1%}"
                )

                c2.metric(
                    "Specificity",
                    f"{baseline_row['Specificity']:.1%}"
                )

                c3.metric(
                    "Balanced accuracy",
                    f"{baseline_row['Balanced accuracy']:.1%}"
                )

            st.markdown("### Download results")

            csv_data = threshold_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download threshold analysis CSV",
                data=csv_data,
                file_name="afrihealth_phase6c_threshold_analysis.csv",
                mime="text/csv",
                key="phase6c_threshold_download"
            )

            st.info(
                "No threshold is automatically recommended. "
                "Any future threshold selection should be based "
                "on a predefined research objective and assessed "
                "without reusing the same cohort as an independent "
                "validation set."
            )

    else:
        st.info(
            "To run this analysis, upload and successfully evaluate "
            "the external validation CSV in the existing External "
            "Validation Workspace above. Then rerun the app."
        )

# ============================================================
# PHASE 6D: EXPLORATORY SUBGROUP PERFORMANCE ANALYSIS
# ============================================================

with st.expander("Phase 6D: Subgroup Analysis", expanded=False):

    st.markdown("""
    ### Exploratory Subgroup Performance Analysis

    This section evaluates whether model performance differs across
    selected subgroups in the external validation cohort.

    The analysis is exploratory and is intended to identify potential
    performance differences that may require further investigation.
    It does not establish clinical fairness, clinical validity, or
    generalisability to any population.
    """)

    # --------------------------------------------------------
    # Required variables
    # --------------------------------------------------------

    required_subgroup_variables = [
        "X_external",
        "y_external",
        "external_probabilities"
    ]

    missing_subgroup_variables = [
        variable
        for variable in required_subgroup_variables
        if variable not in globals()
    ]

    if missing_subgroup_variables:

        st.warning(
            "Subgroup analysis is unavailable because the external "
            "validation results have not been generated in this session."
        )

    else:

        # ----------------------------------------------------
        # Prepare external validation data
        # ----------------------------------------------------

        try:

            subgroup_X = X_external.copy()
            subgroup_y = np.asarray(y_external).astype(int)
            subgroup_probabilities = np.asarray(
                external_probabilities,
                dtype=float
            )

            # Validate row alignment
            if not (
                len(subgroup_X)
                == len(subgroup_y)
                == len(subgroup_probabilities)
            ):
                st.error(
                    "Subgroup analysis stopped because the external "
                    "features, targets, and probabilities have different "
                    "numbers of records."
                )

            else:

                # ------------------------------------------------
                # Create prediction at the frozen 0.50 threshold
                # ------------------------------------------------

                subgroup_predictions = (
                    subgroup_probabilities >= 0.50
                ).astype(int)

                # ------------------------------------------------
                # Helper function for subgroup metrics
                # ------------------------------------------------

                def calculate_subgroup_metrics(
                    y_true_subgroup,
                    y_pred_subgroup,
                    probability_subgroup
                ):

                    from sklearn.metrics import (
                        accuracy_score,
                        balanced_accuracy_score,
                        confusion_matrix,
                        precision_score,
                        recall_score,
                        f1_score,
                        roc_auc_score
                    )

                    n_records = len(y_true_subgroup)

                    if n_records == 0:
                        return None

                    accuracy = accuracy_score(
                        y_true_subgroup,
                        y_pred_subgroup
                    )

                    balanced_accuracy = balanced_accuracy_score(
                        y_true_subgroup,
                        y_pred_subgroup
                    )

                    precision = precision_score(
                        y_true_subgroup,
                        y_pred_subgroup,
                        zero_division=0
                    )

                    sensitivity = recall_score(
                        y_true_subgroup,
                        y_pred_subgroup,
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_true_subgroup,
                        y_pred_subgroup,
                        zero_division=0
                    )

                    cm = confusion_matrix(
                        y_true_subgroup,
                        y_pred_subgroup,
                        labels=[0, 1]
                    )

                    tn, fp, fn, tp = cm.ravel()

                    specificity_denominator = tn + fp

                    if specificity_denominator > 0:
                        specificity = (
                            tn / specificity_denominator
                        )
                    else:
                        specificity = np.nan

                    positive_cases = int(
                        np.sum(y_true_subgroup == 1)
                    )

                    negative_cases = int(
                        np.sum(y_true_subgroup == 0)
                    )

                    # ROC-AUC requires both target classes
                    if len(np.unique(y_true_subgroup)) == 2:
                        roc_auc = roc_auc_score(
                            y_true_subgroup,
                            probability_subgroup
                        )
                    else:
                        roc_auc = np.nan

                    return {
                        "Records": n_records,
                        "Class 0": negative_cases,
                        "Class 1": positive_cases,
                        "Accuracy": accuracy,
                        "Balanced Accuracy": balanced_accuracy,
                        "Sensitivity": sensitivity,
                        "Specificity": specificity,
                        "Precision": precision,
                        "F1": f1,
                        "ROC-AUC": roc_auc,
                        "TN": int(tn),
                        "FP": int(fp),
                        "FN": int(fn),
                        "TP": int(tp)
                    }

                # ------------------------------------------------
                # Build subgroup dataframe
                # ------------------------------------------------

                subgroup_df = subgroup_X.copy()

                subgroup_df["__target"] = subgroup_y
                subgroup_df["__probability"] = (
                    subgroup_probabilities
                )
                subgroup_df["__prediction"] = (
                    subgroup_predictions
                )

                # ------------------------------------------------
                # SEX SUBGROUP ANALYSIS
                # ------------------------------------------------

                st.markdown("### 1. Sex Subgroup Analysis")

                if "sex" in subgroup_df.columns:

                    sex_labels = subgroup_df["sex"].map({
                        0: "Female",
                        1: "Male"
                    })

                    subgroup_df["Sex Group"] = sex_labels

                    sex_results = []

                    for sex_group in ["Female", "Male"]:

                        mask = (
                            subgroup_df["Sex Group"]
                            == sex_group
                        )

                        y_group = subgroup_df.loc[
                            mask,
                            "__target"
                        ].to_numpy()

                        pred_group = subgroup_df.loc[
                            mask,
                            "__prediction"
                        ].to_numpy()

                        probability_group = subgroup_df.loc[
                            mask,
                            "__probability"
                        ].to_numpy()

                        metrics = calculate_subgroup_metrics(
                            y_group,
                            pred_group,
                            probability_group
                        )

                        if metrics is not None:

                            metrics["Subgroup"] = sex_group

                            sex_results.append(metrics)

                    if sex_results:

                        sex_results_df = pd.DataFrame(
                            sex_results
                        )

                        sex_display = sex_results_df[
                            [
                                "Subgroup",
                                "Records",
                                "Class 0",
                                "Class 1",
                                "Accuracy",
                                "Balanced Accuracy",
                                "Sensitivity",
                                "Specificity",
                                "Precision",
                                "F1",
                                "ROC-AUC"
                            ]
                        ].copy()

                        metric_columns = [
                            "Accuracy",
                            "Balanced Accuracy",
                            "Sensitivity",
                            "Specificity",
                            "Precision",
                            "F1",
                            "ROC-AUC"
                        ]

                        sex_display[metric_columns] = (
                            sex_display[metric_columns]
                            .round(3)
                        )

                        st.dataframe(
                            sex_display,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.warning(
                            "No usable sex subgroups were available "
                            "for analysis."
                        )

                else:

                    st.info(
                        "The external validation dataset does not "
                        "contain the required 'sex' feature."
                    )

                # ------------------------------------------------
                # AGE SUBGROUP ANALYSIS
                # ------------------------------------------------

                st.markdown("### 2. Age Subgroup Analysis")

                if "age" in subgroup_df.columns:

                    # Use a transparent and reproducible cutoff.
                    # 55 years is used to provide two broad age groups
                    # without creating excessively small categories.
                    age_cutoff = 55

                    subgroup_df["Age Group"] = np.where(
                        subgroup_df["age"] < age_cutoff,
                        f"Under {age_cutoff}",
                        f"{age_cutoff}+"
                    )

                    age_results = []

                    age_groups = [
                        f"Under {age_cutoff}",
                        f"{age_cutoff}+"
                    ]

                    for age_group in age_groups:

                        mask = (
                            subgroup_df["Age Group"]
                            == age_group
                        )

                        y_group = subgroup_df.loc[
                            mask,
                            "__target"
                        ].to_numpy()

                        pred_group = subgroup_df.loc[
                            mask,
                            "__prediction"
                        ].to_numpy()

                        probability_group = subgroup_df.loc[
                            mask,
                            "__probability"
                        ].to_numpy()

                        metrics = calculate_subgroup_metrics(
                            y_group,
                            pred_group,
                            probability_group
                        )

                        if metrics is not None:

                            metrics["Subgroup"] = age_group

                            age_results.append(metrics)

                    if age_results:

                        age_results_df = pd.DataFrame(
                            age_results
                        )

                        age_display = age_results_df[
                            [
                                "Subgroup",
                                "Records",
                                "Class 0",
                                "Class 1",
                                "Accuracy",
                                "Balanced Accuracy",
                                "Sensitivity",
                                "Specificity",
                                "Precision",
                                "F1",
                                "ROC-AUC"
                            ]
                        ].copy()

                        age_display[metric_columns] = (
                            age_display[metric_columns]
                            .round(3)
                        )

                        st.dataframe(
                            age_display,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:

                        st.warning(
                            "No usable age subgroups were available "
                            "for analysis."
                        )

                else:

                    st.info(
                        "The external validation dataset does not "
                        "contain the required 'age' feature."
                    )

                # ------------------------------------------------
                # SUBGROUP SAMPLE SIZE WARNING
                # ------------------------------------------------

                st.markdown("### 3. Subgroup Interpretation")

                st.info("""
                Subgroup results should be interpreted cautiously.
                Differences in performance may reflect sample size,
                class prevalence, dataset composition, measurement
                differences, or genuine variation in model behaviour.

                Small subgroups can produce unstable estimates.
                Therefore, subgroup results are exploratory and should
                not be interpreted as evidence of clinical fairness,
                discrimination, or population-level validity.
                """)

                # ------------------------------------------------
                # DOWNLOAD SUBGROUP RESULTS
                # ------------------------------------------------

                all_subgroup_results = []

                if "sex_results_df" in locals():

                    sex_export = sex_results_df.copy()
                    sex_export["Subgroup Type"] = "Sex"

                    all_subgroup_results.append(
                        sex_export
                    )

                if "age_results_df" in locals():

                    age_export = age_results_df.copy()
                    age_export["Subgroup Type"] = "Age"

                    all_subgroup_results.append(
                        age_export
                    )

                if all_subgroup_results:

                    subgroup_export = pd.concat(
                        all_subgroup_results,
                        ignore_index=True
                    )

                    csv_subgroup = subgroup_export.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Phase 6D Subgroup Results",
                        data=csv_subgroup,
                        file_name=(
                            "afrihealth_phase6d_subgroup_analysis.csv"
                        ),
                        mime="text/csv"
                    )

                # ------------------------------------------------
                # RESEARCH NOTE
                # ------------------------------------------------

                st.caption(
                    "Phase 6D uses the frozen external-validation "
                    "predictions at a classification threshold of 0.50. "
                    "No model retraining or threshold optimisation is "
                    "performed in this analysis."
                )

        except Exception as e:

            st.error(
                "Phase 6D could not complete because of an "
                "unexpected data or processing error."
            )

            st.exception(e)


# ============================================================
# PHASE 6E: REPRODUCIBILITY AND RESEARCH PROTOCOL
# ============================================================

with st.expander(
    "Phase 6E: Reproducibility and Research Protocol",
    expanded=False
):

    import sys
    import json
    import hashlib
    import platform
    from pathlib import Path
    from datetime import datetime, timezone

    import sklearn

    st.info(
        "This section documents the evaluation configuration, available "
        "software and model artifacts, and known limitations of the "
        "AfriHealth AI research evaluation. It is a documentation and "
        "audit utility. It does not retrain the model, alter predictions, "
        "or establish clinical validity."
    )

    # --------------------------------------------------------
    # 1. FIXED EVALUATION PROTOCOL
    # --------------------------------------------------------

    st.subheader("1. Evaluation Protocol")

    protocol = {
        "project_name": "AfriHealth AI",
        "project_version": "0.1.0",
        "model": "Logistic Regression",
        "dataset_source": "UCI Heart Disease Dataset",
        "uci_repository_id": 45,
        "task": "Binary classification",
        "target_definition": (
            "Class 0: original target equals 0; "
            "Class 1: original target greater than 0"
        ),
        "external_validation_cohort": "Hungarian cohort",
        "classification_threshold": 0.50,
        "threshold_analysis": "Exploratory thresholds from 0.10 to 0.90",
        "age_subgroup_cutoff": 55,
        "age_subgroups": ["Under 55", "55+"],
        "sex_encoding_assumption": {
            "0": "Female",
            "1": "Male"
        },
        "external_validation_frozen": True,
        "model_retraining_during_external_validation": False,
        "intended_use": "Research and educational purposes only",
        "clinical_use_authorized": False
    }

    expected_features = [
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
        "thal"
    ]

    protocol["expected_features"] = expected_features

    protocol_rows = [
        ("Project", protocol["project_name"]),
        ("Version", protocol["project_version"]),
        ("Model", protocol["model"]),
        ("Dataset source", protocol["dataset_source"]),
        ("UCI repository ID", protocol["uci_repository_id"]),
        ("Task", protocol["task"]),
        ("Target definition", protocol["target_definition"]),
        ("External validation cohort", protocol["external_validation_cohort"]),
        ("Classification threshold", protocol["classification_threshold"]),
        ("Threshold analysis", protocol["threshold_analysis"]),
        ("Age subgroup cutoff", protocol["age_subgroup_cutoff"]),
        ("Age subgroups", ", ".join(protocol["age_subgroups"])),
        ("Sex encoding", "0 = Female; 1 = Male"),
        (
            "External validation frozen",
            "Yes" if protocol["external_validation_frozen"] else "No",
        ),
        (
            "Model retraining during external validation",
            "No" if not protocol["model_retraining_during_external_validation"] else "Yes",
        ),
        ("Intended use", protocol["intended_use"]),
        (
            "Clinical use authorized",
            "No" if not protocol["clinical_use_authorized"] else "Yes",
        ),
    ]

    st.dataframe(
        pd.DataFrame(protocol_rows, columns=["Protocol Item", "Value"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Expected input features**")
    st.write(", ".join(expected_features))

    # --------------------------------------------------------
    # 2. SOFTWARE ENVIRONMENT
    # --------------------------------------------------------

    st.subheader("2. Software Environment")

    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform()
    }

    try:
        import streamlit
        environment["streamlit_version"] = streamlit.__version__
    except Exception:
        environment["streamlit_version"] = "Unavailable"

    try:
        import joblib
        environment["joblib_version"] = joblib.__version__
    except Exception:
        environment["joblib_version"] = "Unavailable"

    st.dataframe(
        pd.DataFrame(
            [
                {"Component": key, "Value": value}
                for key, value in environment.items()
            ]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These versions describe the running Streamlit environment. "
        "They do not by themselves prove that the original training "
        "environment has been reproduced."
    )

    # --------------------------------------------------------
    # 3. MODEL AND METADATA FILE INTEGRITY
    # --------------------------------------------------------

    st.subheader("3. Model Artifact Integrity")

    artifact_names = [
        "afrihealth_model.joblib",
        "model_metadata.json"
    ]

    artifact_results = []
    artifact_hashes = {}

    for artifact_name in artifact_names:

        artifact_path = Path(artifact_name)

        if artifact_path.is_file():

            try:
                sha256 = hashlib.sha256()

                with artifact_path.open("rb") as artifact_file:
                    for chunk in iter(
                        lambda: artifact_file.read(1024 * 1024),
                        b""
                    ):
                        sha256.update(chunk)

                digest = sha256.hexdigest()

                artifact_hashes[artifact_name] = digest

                artifact_results.append({
                    "Artifact": artifact_name,
                    "Available": True,
                    "Size (bytes)": artifact_path.stat().st_size,
                    "SHA-256": digest,
                    "Status": "Hash calculated"
                })

            except Exception as artifact_error:

                artifact_results.append({
                    "Artifact": artifact_name,
                    "Available": True,
                    "Size (bytes)": None,
                    "SHA-256": None,
                    "Status": (
                        "Hash failed: "
                        + str(artifact_error)
                    )
                })

        else:

            artifact_results.append({
                "Artifact": artifact_name,
                "Available": False,
                "Size (bytes)": None,
                "SHA-256": None,
                "Status": "File not found in current runtime"
            })

    st.dataframe(
        pd.DataFrame(artifact_results),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "A SHA-256 hash identifies the exact file contents. To verify "
        "that an artifact is unchanged, compare its hash with a "
        "trusted hash recorded previously. A newly calculated hash "
        "alone does not prove authenticity."
    )

    # --------------------------------------------------------
    # 4. READ MODEL METADATA IF AVAILABLE
    # --------------------------------------------------------

    st.subheader("4. Model Metadata")

    metadata_path = Path("model_metadata.json")
    loaded_metadata = {}

    if metadata_path.is_file():

        try:

            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                loaded_metadata = json.load(metadata_file)

            metadata_rows = [
                ("Project", loaded_metadata.get("project", "N/A")),
                ("Version", loaded_metadata.get("version", "N/A")),
                ("Selected model", loaded_metadata.get("selected_model", "N/A")),
                ("Selection metric", loaded_metadata.get("selection_metric", "N/A")),
                (
                    "Cross-validation ROC-AUC",
                    loaded_metadata.get("cross_validation_roc_auc", "N/A"),
                ),
                ("Positive class", loaded_metadata.get("positive_class", "N/A")),
                ("Dataset", loaded_metadata.get("dataset", "N/A")),
                ("Training records", loaded_metadata.get("training_records", "N/A")),
                ("Test records", loaded_metadata.get("test_records", "N/A")),
                (
                    "scikit-learn version",
                    loaded_metadata.get("scikit_learn_version", "N/A"),
                ),
                ("Created at (UTC)", loaded_metadata.get("created_at_utc", "N/A")),
                ("Intended use", loaded_metadata.get("intended_use", "N/A")),
            ]

            st.dataframe(
                pd.DataFrame(metadata_rows, columns=["Metadata Item", "Value"]),
                use_container_width=True,
                hide_index=True,
            )

            metadata_features = loaded_metadata.get("input_features", [])
            if metadata_features:
                st.markdown("**Input features**")
                st.write(", ".join(str(feature) for feature in metadata_features))

        except Exception as metadata_error:

            st.warning(
                "The metadata file exists but could not be parsed: "
                + str(metadata_error)
            )

    else:

        st.info(
            "model_metadata.json was not found in the current runtime. "
            "Metadata cannot be independently checked in this session."
        )

    # --------------------------------------------------------
    # 5. EXTERNAL COHORT SUMMARY
    # --------------------------------------------------------

    st.subheader("5. External Validation Cohort Summary")

    manifest_dataset_summary = {
        "available_in_current_session": False
    }

    if (
        "X_external" in globals()
        and "y_external" in globals()
    ):

        try:

            external_features = X_external.copy()
            external_targets = np.asarray(y_external).reshape(-1)

            if len(external_features) != len(external_targets):

                st.error(
                    "The external feature and target counts do not "
                    "match. Cohort summary was not generated."
                )

            else:

                target_counts = pd.Series(
                    external_targets
                ).value_counts().sort_index()

                target_distribution = {
                    str(key): int(value)
                    for key, value in target_counts.items()
                }

                dataset_summary = {
                    "Records": int(len(external_features)),
                    "Feature columns": int(
                        len(external_features.columns)
                    ),
                    "Class 0 records": int(
                        np.sum(external_targets == 0)
                    ),
                    "Class 1 records": int(
                        np.sum(external_targets == 1)
                    ),
                    "Missing feature values": int(
                        external_features.isna().sum().sum()
                    ),
                    "Duplicate feature rows": int(
                        external_features.duplicated().sum()
                    )
                }

                st.dataframe(
                    pd.DataFrame([
                        {
                            "Measure": key,
                            "Value": value
                        }
                        for key, value in dataset_summary.items()
                    ]),
                    use_container_width=True,
                    hide_index=True
                )

                manifest_dataset_summary = {
                    "available_in_current_session": True,
                    "record_count": int(len(external_features)),
                    "feature_count": int(
                        len(external_features.columns)
                    ),
                    "feature_names": [
                        str(column)
                        for column in external_features.columns
                    ],
                    "target_class_counts": target_distribution,
                    "missing_feature_values": int(
                        external_features.isna().sum().sum()
                    ),
                    "duplicate_feature_rows": int(
                        external_features.duplicated().sum()
                    )
                }

                if list(external_features.columns) != expected_features:

                    st.warning(
                        "The external feature columns or their order "
                        "differ from the documented expected feature "
                        "list. Confirm the model input order before "
                        "interpreting this audit."
                    )

                else:

                    st.success(
                        "External feature names and ordering match "
                        "the documented feature list."
                    )

        except Exception as cohort_error:

            st.warning(
                "The cohort summary could not be generated: "
                + str(cohort_error)
            )

    else:

        st.info(
            "Run external validation in this session to populate the "
            "cohort summary. This section does not invent missing results."
        )

    # --------------------------------------------------------
    # 6. REPRODUCIBILITY CHECKLIST
    # --------------------------------------------------------

    st.subheader("6. Research Reproducibility Checklist")

    checklist_items = [
        "Record the exact model artifact and its SHA-256 hash.",
        "Preserve the original training dataset and preprocessing procedure.",
        "Record the train/test split method and random seed from the original training code.",
        "Record all package versions used during model training.",
        "Preserve the external cohort source, cleaning rules, and inclusion criteria.",
        "Keep external validation separate from model fitting and threshold selection.",
        "Record the classification threshold before interpreting the metrics.",
        "Document missing data, duplicates, class imbalance, and constant features.",
        "Report subgroup sizes and avoid unsupported fairness claims.",
        "Retain the evaluation results, code version, and limitations."
    ]

    checklist_status = {}

    for index, checklist_item in enumerate(checklist_items):

        checklist_status[checklist_item] = st.checkbox(
            checklist_item,
            key=f"phase6e_check_{index}"
        )

    completed_checks = sum(checklist_status.values())
    total_checks = len(checklist_items)

    st.progress(completed_checks / total_checks)

    st.write(
        f"Checklist completed in this session: "
        f"{completed_checks} of {total_checks}"
    )

    st.caption(
        "Checklist selections are not persisted as an audit record. "
        "They must not be interpreted as proof that an independent "
        "verification has taken place."
    )

    # --------------------------------------------------------
    # 7. KNOWN LIMITATIONS
    # --------------------------------------------------------

    st.subheader("7. Known Research Limitations")

    known_limitations = [
        (
            "The model was developed using a small historical UCI "
            "Heart Disease dataset."
        ),
        (
            "The Hungarian cohort is an external dataset, but it is "
            "not a substitute for validation in a contemporary "
            "Nigerian or African clinical population."
        ),
        (
            "The external cohort's ca feature was reported as constant "
            "at zero in the existing data-quality audit. The cause "
            "was not independently verified."
        ),
        (
            "A duplicate row was reported in the external cohort. "
            "It was not automatically removed."
        ),
        (
            "Subgroup analyses are exploratory and may be affected "
            "by small sample sizes and class prevalence."
        ),
        (
            "Performance metrics do not establish clinical utility, "
            "calibration in a new population, or clinical safety."
        )
    ]

    for limitation in known_limitations:
        st.markdown(f"- {limitation}")

    # --------------------------------------------------------
    # 8. BUILD DOWNLOADABLE RESEARCH MANIFEST
    # --------------------------------------------------------

    st.subheader("8. Download Research Manifest")

    research_manifest = {
        "manifest_schema_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": "AfriHealth AI",
        "protocol": protocol,
        "runtime_environment": environment,
        "artifact_sha256": artifact_hashes,
        "model_metadata": loaded_metadata,
        "external_dataset_summary": manifest_dataset_summary,
        "reproducibility_checklist": checklist_status,
        "known_limitations": known_limitations,
        "interpretation": {
            "research_only": True,
            "clinical_diagnosis": False,
            "clinical_validation_claimed": False,
            "subgroup_fairness_claimed": False
        }
    }

    manifest_json = json.dumps(
        research_manifest,
        indent=2,
        ensure_ascii=False,
        default=str
    )

    st.download_button(
        label="Download Phase 6E Research Manifest (JSON)",
        data=manifest_json.encode("utf-8"),
        file_name="afrihealth_phase6e_research_manifest.json",
        mime="application/json",
        key="phase6e_manifest_download"
    )

    st.success(
        "Phase 6E research protocol interface is ready. "
        "Review the artifact and cohort checks before treating the "
        "manifest as a completed reproducibility record."
    )

    st.caption(
        "The manifest records the current runtime. It does not replace "
        "the original training code, a locked dependency file, the "
        "raw-data provenance record, or an independently reproduced "
        "evaluation."
    )
# ============================================================
# PHASE 6F: RESEARCH-READY VALIDATION REPORT
# ============================================================

with st.expander(
    "Phase 6F: Research-Ready Validation Report",
    expanded=False
):

    st.subheader("Research-Ready Validation Report")

    st.markdown("""
    This section converts the completed AfriHealth AI evaluation
    activities into a structured research-oriented validation report.

    The report summarizes model development information, external
    validation performance, exploratory threshold analysis, subgroup
    analysis, reproducibility information, and known limitations.

    This report is intended for research and educational documentation.
    It does not establish clinical validity, clinical safety, or
    suitability for diagnosis or treatment.
    """)

    # --------------------------------------------------------
    # 1. CHECK REQUIRED EXTERNAL VALIDATION OBJECTS
    # --------------------------------------------------------

    required_report_variables = [
        "y_external",
        "external_predictions",
        "external_probabilities",
        "external_metrics"
    ]

    missing_report_variables = [
        variable
        for variable in required_report_variables
        if variable not in globals()
    ]

    if missing_report_variables:

        st.warning(
            "Phase 6F cannot generate the validation report because "
            "the required external validation results are not available "
            "in the current session."
        )

        st.caption(
            "Run the External Validation Workspace successfully first, "
            "then return to Phase 6F."
        )

    else:

        try:

            # ------------------------------------------------
            # 2. PREPARE VALIDATION DATA
            # ------------------------------------------------

            report_y = np.asarray(
                y_external
            ).reshape(-1).astype(int)

            report_predictions = np.asarray(
                external_predictions
            ).reshape(-1).astype(int)

            report_probabilities = np.asarray(
                external_probabilities
            ).reshape(-1).astype(float)

            report_n = len(report_y)

            validation_data_valid = (
                report_n > 0
                and len(report_predictions) == report_n
                and len(report_probabilities) == report_n
                and set(np.unique(report_y)).issubset({0, 1})
                and set(np.unique(report_predictions)).issubset({0, 1})
                and np.isfinite(report_probabilities).all()
                and (
                    (report_probabilities >= 0)
                    & (report_probabilities <= 1)
                ).all()
            )

            if not validation_data_valid:

                st.error(
                    "Phase 6F stopped because the external validation "
                    "data failed the required consistency checks."
                )

            else:

                # ------------------------------------------------
                # 3. REPORT HEADER
                # ------------------------------------------------

                report_timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                st.success(
                    "Research-ready validation report generated "
                    "from the frozen external-validation results."
                )

                st.caption(
                    f"Report generated: {report_timestamp}"
                )

                # ------------------------------------------------
                # 4. RESEARCH SUMMARY
                # ------------------------------------------------

                st.markdown("### 1. Research Summary")

                st.write(
                    f"""
                    AfriHealth AI is an AI-assisted cardiovascular
                    disease risk screening research prototype based on
                    a Logistic Regression model trained using the UCI
                    Heart Disease dataset.

                    The selected model was evaluated on an external
                    Hungarian cohort containing {report_n} records.

                    The external validation was performed using the
                    frozen model without model retraining. The default
                    classification threshold was 0.50.
                    """
                )

                st.info(
                    """
                    Primary research interpretation:

                    The external evaluation provides evidence that the
                    model retains measurable predictive discrimination
                    on an independent historical cohort. However, the
                    evaluation does not establish clinical validity,
                    clinical safety, population-level generalisability,
                    or suitability for deployment in Nigerian or other
                    African clinical settings.
                    """
                )

                # ------------------------------------------------
                # 5. MODEL AND DATASET DESCRIPTION
                # ------------------------------------------------

                st.markdown("### 2. Model and Dataset")

                report_model = metadata.get(
                    "selected_model",
                    "Logistic Regression"
                )

                report_version = metadata.get(
                    "version",
                    metadata.get(
                        "project_version",
                        "0.1.0"
                    )
                )

                report_cv_auc = metadata.get(
                    "cross_validation_roc_auc",
                    metadata.get(
                        "cv_roc_auc",
                        None
                    )
                )

                report_train_records = metadata.get(
                    "train_records",
                    "N/A"
                )

                report_holdout_records = metadata.get(
                    "test_records",
                    metadata.get(
                        "holdout_records",
                        "N/A"
                    )
                )

                report_dataset = metadata.get(
                    "dataset",
                    "UCI Heart Disease Dataset"
                )

                report_model_table = pd.DataFrame(
                    [
                        [
                            "Project",
                            "AfriHealth AI"
                        ],
                        [
                            "Version",
                            report_version
                        ],
                        [
                            "Selected model",
                            report_model
                        ],
                        [
                            "Dataset",
                            report_dataset
                        ],
                        [
                            "UCI repository ID",
                            45
                        ],
                        [
                            "Task",
                            "Binary classification"
                        ],
                        [
                            "Training records",
                            report_train_records
                        ],
                        [
                            "Internal holdout records",
                            report_holdout_records
                        ],
                        [
                            "External validation cohort",
                            "Hungarian cohort"
                        ],
                        [
                            "External validation records",
                            report_n
                        ],
                        [
                            "Classification threshold",
                            0.50
                        ]
                    ],
                    columns=[
                        "Item",
                        "Value"
                    ]
                )

                st.dataframe(
                    report_model_table,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # 6. DEVELOPMENT EVIDENCE
                # ------------------------------------------------

                st.markdown("### 3. Development Evidence")

                development_rows = []

                if report_cv_auc is not None:

                    try:
                        development_rows.append(
                            [
                                "Cross-validation ROC-AUC",
                                f"{float(report_cv_auc):.3f}"
                            ]
                        )
                    except Exception:

                        development_rows.append(
                            [
                                "Cross-validation ROC-AUC",
                                str(report_cv_auc)
                            ]
                        )

                development_rows.extend(
                    [
                        [
                            "Selected model",
                            str(report_model)
                        ],
                        [
                            "Training records",
                            str(report_train_records)
                        ],
                        [
                            "Internal holdout records",
                            str(report_holdout_records)
                        ],
                        [
                            "External validation retraining",
                            "No"
                        ],
                        [
                            "External validation threshold",
                            "0.50"
                        ]
                    ]
                )

                development_df = pd.DataFrame(
                    development_rows,
                    columns=[
                        "Development Evidence",
                        "Value"
                    ]
                )

                st.dataframe(
                    development_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # 7. EXTERNAL VALIDATION PERFORMANCE
                # ------------------------------------------------

                st.markdown("### 4. External Validation Performance")

                metric_order = [
                    "Accuracy",
                    "Balanced Accuracy",
                    "Sensitivity",
                    "Specificity",
                    "Precision",
                    "F1 Score",
                    "ROC-AUC",
                    "Brier Score"
                ]

                external_metric_rows = []

                for metric_name in metric_order:

                    if metric_name in external_metrics:

                        value = external_metrics[
                            metric_name
                        ]

                        try:
                            value_display = f"{float(value):.3f}"
                        except Exception:
                            value_display = str(value)

                        external_metric_rows.append(
                            [
                                metric_name,
                                value_display
                            ]
                        )

                external_metric_df = pd.DataFrame(
                    external_metric_rows,
                    columns=[
                        "Metric",
                        "External Validation"
                    ]
                )

                st.dataframe(
                    external_metric_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # 8. KEY PERFORMANCE METRICS
                # ------------------------------------------------

                key_cols = st.columns(4)

                with key_cols[0]:
                    st.metric(
                        "External ROC-AUC",
                        f"{external_metrics['ROC-AUC']:.3f}"
                    )

                with key_cols[1]:
                    st.metric(
                        "Sensitivity",
                        f"{external_metrics['Sensitivity']:.3f}"
                    )

                with key_cols[2]:
                    st.metric(
                        "Specificity",
                        f"{external_metrics['Specificity']:.3f}"
                    )

                with key_cols[3]:
                    st.metric(
                        "Brier Score",
                        f"{external_metrics['Brier Score']:.3f}"
                    )

                # ------------------------------------------------
                # 9. BOOTSTRAP CONFIDENCE INTERVALS
                # ------------------------------------------------

                st.markdown(
                    "### 5. Bootstrap 95% Confidence Intervals"
                )

                st.caption(
                    "Confidence intervals are estimated from the frozen "
                    "external-validation predictions using a fixed "
                    "bootstrap seed. They are descriptive research "
                    "estimates and do not compensate for dataset "
                    "representativeness limitations."
                )

                from sklearn.metrics import (
                    accuracy_score,
                    balanced_accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    roc_auc_score,
                    brier_score_loss
                )

                rng = np.random.default_rng(42)

                bootstrap_iterations = 2000

                bootstrap_values = {
                    "Accuracy": [],
                    "Balanced Accuracy": [],
                    "Sensitivity": [],
                    "Specificity": [],
                    "Precision": [],
                    "F1 Score": [],
                    "ROC-AUC": [],
                    "Brier Score": []
                }

                for _ in range(
                    bootstrap_iterations
                ):

                    sample_indices = rng.integers(
                        0,
                        report_n,
                        report_n
                    )

                    y_b = report_y[
                        sample_indices
                    ]

                    pred_b = report_predictions[
                        sample_indices
                    ]

                    prob_b = report_probabilities[
                        sample_indices
                    ]

                    try:

                        tn, fp, fn, tp = confusion_matrix(
                            y_b,
                            pred_b,
                            labels=[0, 1]
                        ).ravel()

                        specificity_b = (
                            tn / (tn + fp)
                            if (tn + fp) > 0
                            else np.nan
                        )

                        bootstrap_values[
                            "Accuracy"
                        ].append(
                            accuracy_score(
                                y_b,
                                pred_b
                            )
                        )

                        bootstrap_values[
                            "Balanced Accuracy"
                        ].append(
                            balanced_accuracy_score(
                                y_b,
                                pred_b
                            )
                        )

                        bootstrap_values[
                            "Sensitivity"
                        ].append(
                            recall_score(
                                y_b,
                                pred_b,
                                zero_division=0
                            )
                        )

                        bootstrap_values[
                            "Specificity"
                        ].append(
                            specificity_b
                        )

                        bootstrap_values[
                            "Precision"
                        ].append(
                            precision_score(
                                y_b,
                                pred_b,
                                zero_division=0
                            )
                        )

                        bootstrap_values[
                            "F1 Score"
                        ].append(
                            f1_score(
                                y_b,
                                pred_b,
                                zero_division=0
                            )
                        )

                        if len(
                            np.unique(y_b)
                        ) == 2:

                            bootstrap_values[
                                "ROC-AUC"
                            ].append(
                                roc_auc_score(
                                    y_b,
                                    prob_b
                                )
                            )

                        else:

                            bootstrap_values[
                                "ROC-AUC"
                            ].append(
                                np.nan
                            )

                        bootstrap_values[
                            "Brier Score"
                        ].append(
                            brier_score_loss(
                                y_b,
                                prob_b
                            )
                        )

                    except Exception:

                        continue

                confidence_rows = []

                for metric_name in metric_order:

                    values = np.asarray(
                        bootstrap_values[
                            metric_name
                        ],
                        dtype=float
                    )

                    values = values[
                        np.isfinite(values)
                    ]

                    if len(values) > 20:

                        lower = np.percentile(
                            values,
                            2.5
                        )

                        upper = np.percentile(
                            values,
                            97.5
                        )

                        point_value = external_metrics[
                            metric_name
                        ]

                        confidence_rows.append(
                            [
                                metric_name,
                                f"{float(point_value):.3f}",
                                f"{lower:.3f}",
                                f"{upper:.3f}",
                                len(values)
                            ]
                        )

                confidence_df = pd.DataFrame(
                    confidence_rows,
                    columns=[
                        "Metric",
                        "Point Estimate",
                        "95% CI Lower",
                        "95% CI Upper",
                        "Bootstrap Samples"
                    ]
                )

                st.dataframe(
                    confidence_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # 10. VALIDATION INTERPRETATION
                # ------------------------------------------------

                st.markdown("### 6. Validation Interpretation")

                auc_value = float(
                    external_metrics["ROC-AUC"]
                )

                sensitivity_value = float(
                    external_metrics["Sensitivity"]
                )

                specificity_value = float(
                    external_metrics["Specificity"]
                )

                brier_value = float(
                    external_metrics["Brier Score"]
                )

                interpretation_points = []

                if auc_value >= 0.80:

                    interpretation_points.append(
                        "The external cohort shows substantial "
                        "discriminative performance based on ROC-AUC."
                    )

                elif auc_value >= 0.70:

                    interpretation_points.append(
                        "The external cohort shows moderate "
                        "discriminative performance based on ROC-AUC."
                    )

                else:

                    interpretation_points.append(
                        "The external cohort shows limited "
                        "discriminative performance based on ROC-AUC."
                    )

                interpretation_points.append(
                    f"Sensitivity at the frozen 0.50 threshold was "
                    f"{sensitivity_value:.3f}."
                )

                interpretation_points.append(
                    f"Specificity at the frozen 0.50 threshold was "
                    f"{specificity_value:.3f}."
                )

                interpretation_points.append(
                    f"The observed Brier score was "
                    f"{brier_value:.3f}. This provides a measure of "
                    f"probabilistic prediction error but does not by "
                    f"itself establish clinical calibration."
                )

                interpretation_points.append(
                    "The external cohort provides useful evidence for "
                    "research evaluation but cannot establish "
                    "performance in Nigerian or broader African "
                    "populations."
                )

                for point in interpretation_points:

                    st.markdown(
                        f"- {point}"
                    )

                # ------------------------------------------------
                # 11. THRESHOLD ANALYSIS SUMMARY
                # ------------------------------------------------

                st.markdown(
                    "### 7. Threshold Analysis Interpretation"
                )

                st.write(
                    """
                    Phase 6C evaluated classification thresholds from
                    0.10 to 0.90 using the frozen external-validation
                    probabilities.

                    The threshold analysis is exploratory. It should
                    not be interpreted as a basis for selecting a
                    clinically optimal threshold because the same
                    external cohort was used for evaluation.
                    """
                )

                threshold_results_available = (
                    "threshold_df" in globals()
                    and isinstance(
                        threshold_df,
                        pd.DataFrame
                    )
                    and not threshold_df.empty
                )

                if threshold_results_available:

                    threshold_display = threshold_df.copy()

                    threshold_metric_columns = [
                        column
                        for column in [
                            "Accuracy",
                            "Balanced accuracy",
                            "Sensitivity",
                            "Specificity",
                            "Precision",
                            "F1 score"
                        ]
                        if column in threshold_display.columns
                    ]

                    if threshold_metric_columns:

                        threshold_display[
                            threshold_metric_columns
                        ] = threshold_display[
                            threshold_metric_columns
                        ].round(3)

                    st.dataframe(
                        threshold_display,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "Threshold analysis results were not available "
                        "in the current session."
                    )

                # ------------------------------------------------
                # 12. SUBGROUP ANALYSIS SUMMARY
                # ------------------------------------------------

                st.markdown(
                    "### 8. Subgroup Analysis Interpretation"
                )

                st.write(
                    """
                    Phase 6D examined selected sex and age subgroups
                    using the frozen external-validation predictions
                    at the 0.50 classification threshold.

                    These analyses are exploratory. Differences between
                    subgroup estimates may reflect sample size, class
                    prevalence, dataset composition, measurement
                    differences, or genuine differences in model
                    behaviour.
                    """
                )

                subgroup_results_available = (
                    (
                        "sex_results_df" in globals()
                        and isinstance(
                            sex_results_df,
                            pd.DataFrame
                        )
                    )
                    or
                    (
                        "age_results_df" in globals()
                        and isinstance(
                            age_results_df,
                            pd.DataFrame
                        )
                    )
                )

                if subgroup_results_available:

                    st.success(
                        "Subgroup analysis results are available "
                        "from Phase 6D."
                    )

                    if (
                        "sex_results_df" in globals()
                        and isinstance(
                            sex_results_df,
                            pd.DataFrame
                        )
                        and not sex_results_df.empty
                    ):

                        st.markdown("**Sex subgroup results**")

                        st.dataframe(
                            sex_results_df.round(3),
                            use_container_width=True,
                            hide_index=True
                        )

                    if (
                        "age_results_df" in globals()
                        and isinstance(
                            age_results_df,
                            pd.DataFrame
                        )
                        and not age_results_df.empty
                    ):

                        st.markdown("**Age subgroup results**")

                        st.dataframe(
                            age_results_df.round(3),
                            use_container_width=True,
                            hide_index=True
                        )

                else:

                    st.info(
                        "No subgroup result tables are available "
                        "in the current session."
                    )

                # ------------------------------------------------
                # 13. DATA QUALITY FINDINGS
                # ------------------------------------------------

                st.markdown(
                    "### 9. External Cohort Data-Quality Findings"
                )

                data_quality_rows = [
                    [
                        "External cohort",
                        "Hungarian cohort"
                    ],
                    [
                        "Records evaluated",
                        report_n
                    ],
                    [
                        "Missing values",
                        "None reported in Phase 6B"
                    ],
                    [
                        "ca feature",
                        "Constant at zero"
                    ],
                    [
                        "Duplicate row",
                        "One duplicate reported"
                    ],
                    [
                        "External cohort used for retraining",
                        "No"
                    ]
                ]

                data_quality_df = pd.DataFrame(
                    data_quality_rows,
                    columns=[
                        "Data Quality Item",
                        "Finding"
                    ]
                )

                st.dataframe(
                    data_quality_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.warning(
                    """
                    The constant `ca` feature and reported duplicate
                    record are important limitations of the external
                    cohort. These findings should be disclosed in any
                    research manuscript or presentation using these
                    results.
                    """
                )

                # ------------------------------------------------
                # 14. REPRODUCIBILITY STATUS
                # ------------------------------------------------

                st.markdown(
                    "### 10. Reproducibility Status"
                )

                reproducibility_rows = [
                    [
                        "Model artifact available",
                        "Yes"
                    ],
                    [
                        "Model metadata available",
                        "Yes"
                    ],
                    [
                        "External validation frozen",
                        "Yes"
                    ],
                    [
                        "Model retrained during validation",
                        "No"
                    ],
                    [
                        "Classification threshold documented",
                        "Yes, 0.50"
                    ],
                    [
                        "Threshold analysis documented",
                        "Yes"
                    ],
                    [
                        "Subgroup analysis documented",
                        "Yes"
                    ],
                    [
                        "Research manifest available",
                        "Yes, Phase 6E"
                    ],
                    [
                        "Independent Nigerian/African validation",
                        "Not performed"
                    ]
                ]

                reproducibility_df = pd.DataFrame(
                    reproducibility_rows,
                    columns=[
                        "Reproducibility Item",
                        "Status"
                    ]
                )

                st.dataframe(
                    reproducibility_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # 15. RESEARCH LIMITATIONS
                # ------------------------------------------------

                st.markdown(
                    "### 11. Research Limitations"
                )

                report_limitations = [
                    (
                        "The development dataset is relatively small "
                        "and originates from the historical UCI Heart "
                        "Disease dataset."
                    ),
                    (
                        "The external validation cohort is historical "
                        "and does not represent a contemporary Nigerian "
                        "or African clinical population."
                    ),
                    (
                        "The external cohort contains a constant "
                        "`ca` feature, limiting evaluation of that "
                        "predictor."
                    ),
                    (
                        "One duplicate record was reported in the "
                        "external cohort and was not automatically "
                        "removed."
                    ),
                    (
                        "Subgroup analyses are exploratory and may "
                        "contain unstable estimates because of subgroup "
                        "sample sizes."
                    ),
                    (
                        "Threshold analysis is exploratory and was "
                        "performed on the same external cohort used "
                        "for validation."
                    ),
                    (
                        "The evaluation does not establish clinical "
                        "utility, clinical safety, or clinical "
                        "calibration."
                    ),
                    (
                        "No validation has been performed on a "
                        "representative Nigerian or African clinical "
                        "population."
                    )
                ]

                for limitation in report_limitations:

                    st.markdown(
                        f"- {limitation}"
                    )

                # ------------------------------------------------
                # 16. RESPONSIBLE AI INTERPRETATION
                # ------------------------------------------------

                st.markdown(
                    "### 12. Responsible AI Interpretation"
                )

                st.info(
                    """
                    AfriHealth AI should currently be interpreted as
                    an educational and research prototype.

                    The results should not be used to diagnose disease,
                    determine treatment, replace clinicians, or make
                    autonomous medical decisions.

                    Before any real-world clinical consideration,
                    further validation would be required using
                    appropriately designed contemporary datasets,
                    including representative Nigerian and African
                    populations, external calibration assessment,
                    prospective evaluation, clinical review, and
                    appropriate regulatory and ethical oversight.
                    """
                )

                # ------------------------------------------------
                # 17. OVERALL VALIDATION CONCLUSION
                # ------------------------------------------------

                st.markdown(
                    "### 13. Overall Validation Conclusion"
                )

                st.success(
                    f"""
                    The AfriHealth AI Logistic Regression model achieved
                    an external ROC-AUC of {auc_value:.3f} on the
                    evaluated Hungarian cohort.

                    The external evaluation demonstrates measurable
                    predictive discrimination outside the original
                    development data and therefore provides useful
                    research evidence for continued investigation.

                    However, the available evidence is not sufficient
                    to claim clinical validity, clinical safety,
                    population-level generalisability, or suitability
                    for deployment in Nigerian or African healthcare
                    settings.

                    The appropriate next step is broader independent
                    validation using representative contemporary
                    populations and a predefined research protocol.
                    """
                )

                # ------------------------------------------------
                # 18. BUILD DOWNLOADABLE MARKDOWN REPORT
                # ------------------------------------------------

                st.markdown(
                    "### 14. Download Research Report"
                )

                markdown_lines = [
                    "# AfriHealth AI Research-Ready Validation Report",
                    "",
                    f"Report generated: {report_timestamp}",
                    "",
                    "## 1. Research Summary",
                    "",
                    (
                        "AfriHealth AI is an AI-assisted cardiovascular "
                        "disease risk screening research prototype "
                        "based on Logistic Regression and the UCI Heart "
                        "Disease dataset."
                    ),
                    "",
                    (
                        f"External validation was performed on "
                        f"{report_n} records from the Hungarian cohort "
                        "using the frozen model and a 0.50 classification "
                        "threshold."
                    ),
                    "",
                    "## 2. Model and Dataset",
                    "",
                    f"- Project: AfriHealth AI",
                    f"- Version: {report_version}",
                    f"- Model: {report_model}",
                    f"- Dataset: {report_dataset}",
                    f"- UCI repository ID: 45",
                    f"- Training records: {report_train_records}",
                    f"- Internal holdout records: {report_holdout_records}",
                    f"- External validation records: {report_n}",
                    f"- External validation retraining: No",
                    f"- Classification threshold: 0.50",
                    "",
                    "## 3. External Validation Performance",
                    ""
                ]

                for metric_name in metric_order:

                    if metric_name in external_metrics:

                        markdown_lines.append(
                            f"- {metric_name}: "
                            f"{float(external_metrics[metric_name]):.3f}"
                        )

                markdown_lines.extend(
                    [
                        "",
                        "## 4. Bootstrap 95% Confidence Intervals",
                        ""
                    ]
                )

                for row in confidence_rows:

                    markdown_lines.append(
                        f"- {row[0]}: {row[1]} "
                        f"(95% CI {row[2]}–{row[3]})"
                    )

                markdown_lines.extend(
                    [
                        "",
                        "## 5. Threshold Analysis",
                        "",
                        (
                            "Threshold analysis from 0.10 to 0.90 "
                            "was exploratory and used the frozen "
                            "external-validation predictions."
                        ),
                        "",
                        "## 6. Subgroup Analysis",
                        "",
                        (
                            "Sex and age subgroup analyses were "
                            "exploratory and were performed using "
                            "the frozen external-validation results."
                        ),
                        "",
                        "## 7. Data-Quality Findings",
                        "",
                        "- No missing values were reported in the external cohort.",
                        "- The `ca` feature was constant at zero.",
                        "- One duplicate row was reported.",
                        "- The model was not retrained on the external cohort.",
                        "",
                        "## 8. Reproducibility",
                        "",
                        "- Model artifact available.",
                        "- Model metadata available.",
                        "- External validation frozen.",
                        "- Threshold documented.",
                        "- Threshold analysis documented.",
                        "- Subgroup analysis documented.",
                        "- Phase 6E research manifest available.",
                        "",
                        "## 9. Limitations",
                        ""
                    ]
                )

                for limitation in report_limitations:

                    markdown_lines.append(
                        f"- {limitation}"
                    )

                markdown_lines.extend(
                    [
                        "",
                        "## 10. Responsible Interpretation",
                        "",
                        (
                            "AfriHealth AI is a research and educational "
                            "prototype. It is not a medical diagnostic "
                            "system and should not be used to make "
                            "autonomous clinical decisions."
                        ),
                        "",
                        "## 11. Overall Conclusion",
                        "",
                        (
                            f"The model achieved an external ROC-AUC "
                            f"of {auc_value:.3f} on the evaluated "
                            "Hungarian cohort. The result provides "
                            "useful evidence for continued research "
                            "but does not establish clinical validity "
                            "or generalisability to Nigerian or "
                            "African populations."
                        ),
                        ""
                    ]
                )

                markdown_report = "\n".join(
                    markdown_lines
                )

                st.download_button(
                    label=(
                        "Download Phase 6F Research Report "
                        "(Markdown)"
                    ),
                    data=markdown_report.encode("utf-8"),
                    file_name=(
                        "afrihealth_phase6f_validation_report.md"
                    ),
                    mime="text/markdown",
                    key="phase6f_markdown_report"
                )

                # ------------------------------------------------
                # 19. DOWNLOAD STRUCTURED JSON SUMMARY
                # ------------------------------------------------

                research_summary = {
                    "report_schema_version": "1.0",
                    "generated_at_utc": report_timestamp,
                    "project": "AfriHealth AI",
                    "project_version": str(report_version),
                    "model": str(report_model),
                    "dataset": str(report_dataset),
                    "uci_repository_id": 45,
                    "external_validation": {
                        "cohort": "Hungarian cohort",
                        "records": int(report_n),
                        "classification_threshold": 0.50,
                        "model_retraining": False,
                        "metrics": {
                            key: float(value)
                            for key, value in external_metrics.items()
                            if isinstance(
                                value,
                                (int, float, np.integer, np.floating)
                            )
                        }
                    },
                    "bootstrap": {
                        "iterations": bootstrap_iterations,
                        "random_seed": 42,
                        "confidence_level": 0.95
                    },
                    "limitations": report_limitations,
                    "interpretation": {
                        "research_only": True,
                        "clinical_diagnosis": False,
                        "clinical_validation_claimed": False,
                        "nigerian_population_validation": False,
                        "african_population_validation": False
                    }
                }

                research_summary_json = json.dumps(
                    research_summary,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )

                st.download_button(
                    label=(
                        "Download Phase 6F Research Summary "
                        "(JSON)"
                    ),
                    data=research_summary_json.encode(
                        "utf-8"
                    ),
                    file_name=(
                        "afrihealth_phase6f_validation_summary.json"
                    ),
                    mime="application/json",
                    key="phase6f_json_report"
                )

                # ------------------------------------------------
                # 20. RESEARCH STATUS
                # ------------------------------------------------

                st.divider()

                st.success(
                    """
                    Phase 6F complete.

                    AfriHealth AI now has a consolidated
                    research-ready validation report covering
                    external validation, confidence intervals,
                    threshold analysis, subgroup analysis,
                    data-quality findings, reproducibility,
                    limitations, and responsible interpretation.
                    """
                )

                st.caption(
                    "Phase 6F is a reporting and documentation layer. "
                    "It does not retrain the model, modify predictions, "
                    "or establish clinical validity."
                )

        except Exception as phase6f_error:

            st.error(
                "Phase 6F could not generate the research-ready "
                "validation report."
            )

            st.exception(phase6f_error)
# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#6b7280;">
    <b>AfriHealth AI</b><br>
    AI-assisted heart disease prediction research prototype<br>
    Research and educational use only. Not for diagnosis or treatment.
    </div>
    """,
    unsafe_allow_html=True,
)
