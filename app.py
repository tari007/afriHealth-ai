
import json
from io import BytesIO
from pathlib import Path

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
# 1. CONFIGURATION
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
EVALUATION_PATH = BASE_DIR / "evaluation_predictions.csv"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]

FEATURE_DESCRIPTIONS = [
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
]

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .subtitle {
        color: #16c7b7;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. LOAD MODEL AND METADATA
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "afrihealth_model.joblib is missing from the repository root."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.is_file():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    metadata = load_metadata()
except Exception as error:
    st.error("AfriHealth AI could not load its model.")
    st.code(str(error))
    st.stop()

if not hasattr(model, "predict"):
    st.error("The saved model does not support predictions.")
    st.stop()

model_features = list(
    metadata.get("input_features", FEATURES)
)

if model_features != FEATURES:
    st.error(
        "The input_features list in model_metadata.json does not "
        "match the feature order expected by this application."
    )
    st.stop()

try:
    model_classes = list(model.classes_)
except Exception:
    model_classes = []

if not {0, 1}.issubset(set(model_classes)):
    st.error(
        "The saved model must support binary classes 0 and 1."
    )
    st.stop()


# ============================================================
# 3. SHARED HELPER FUNCTIONS
# ============================================================

def safe_metric(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall / Sensitivity": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1 score": f1_score(
            actual, predicted, zero_division=0
        ),
        "Balanced accuracy": balanced_accuracy_score(
            actual, predicted
        ),
    }


def validate_evaluation_data(data):
    required = {"y_true", "y_pred", "prob_class_1"}

    if not required.issubset(data.columns):
        missing = required - set(data.columns)
        raise ValueError(
            "Missing evaluation columns: "
            + ", ".join(sorted(missing))
        )

    if data.empty:
        raise ValueError("The evaluation CSV is empty.")

    actual = pd.to_numeric(
        data["y_true"], errors="raise"
    ).to_numpy(dtype=float)

    predicted = pd.to_numeric(
        data["y_pred"], errors="raise"
    ).to_numpy(dtype=float)

    probabilities = pd.to_numeric(
        data["prob_class_1"], errors="raise"
    ).to_numpy(dtype=float)

    if not (
        len(actual) == len(predicted) == len(probabilities)
    ):
        raise ValueError("Evaluation columns have different lengths.")

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
        and np.isfinite(probabilities).all()
    ):
        raise ValueError("Evaluation data contains invalid numbers.")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Actual labels must be 0 or 1.")

    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Predicted labels must be 0 or 1.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    return actual.astype(int), predicted.astype(int), probabilities


def make_confusion_matrix_figure(actual, predicted):
    matrix = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[
                [0, "#102235"],
                [0.5, "#087f8c"],
                [1, "#5fffe9"],
            ],
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            showscale=False,
        )
    )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Predicted class",
        yaxis_title="Actual class",
        height=380,
    )
    return figure


def make_roc_figure(actual, probabilities, title="ROC curve"):
    false_positive, true_positive, _ = roc_curve(
        actual, probabilities, pos_label=1
    )
    auc_value = roc_auc_score(actual, probabilities)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=false_positive,
            y=true_positive,
            mode="lines",
            name=f"ROC-AUC = {auc_value:.3f}",
            line={"color": "#5fffe9", "width": 3},
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance reference",
            line={
                "color": "#9fb0c0",
                "width": 2,
                "dash": "dash",
            },
        )
    )

    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.02]},
        height=420,
    )

    return figure, auc_value


def make_calibration_figure(
    actual,
    probabilities,
    bins=5,
    strategy="quantile",
    title="Probability calibration",
):
    observed, mean_predicted = calibration_curve(
        actual,
        probabilities,
        n_bins=bins,
        strategy=strategy,
        pos_label=1,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ideal calibration",
            line={"dash": "dash", "color": "#9fb0c0"},
        )
    )

    figure.add_trace(
        go.Scatter(
            x=mean_predicted,
            y=observed,
            mode="lines+markers",
            name="Observed calibration",
            line={"color": "#5fffe9", "width": 3},
            marker={"size": 9},
        )
    )

    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Mean predicted class-1 probability",
        yaxis_title="Observed class-1 fraction",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        height=420,
    )

    return figure


def create_pdf_report(metrics, project_metadata, record_count, report_title):
    # ReportLab is already included in the project's requirements.txt.
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=report_title,
        author="AfriHealth AI",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="AfriTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#087f8c"),
            spaceAfter=12,
        )
    )

    story = [
        Paragraph("AfriHealth AI", styles["AfriTitle"]),
        Paragraph(report_title, styles["Heading2"]),
        Spacer(1, 8),
        Paragraph(
            "Research and educational use only. This report is not "
            "a medical diagnosis and must not be used to make treatment "
            "or clinical decisions.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Project Information", styles["Heading2"]),
    ]

    project_rows = [
        ["Project", "AfriHealth AI"],
        [
            "Model",
            str(project_metadata.get("selected_model", "Not recorded")),
        ],
        [
            "Dataset",
            str(
                project_metadata.get(
                    "dataset", "UCI Heart Disease dataset"
                )
            ),
        ],
        ["Evaluation records", str(record_count)],
        ["Evaluation basis", "Provided evaluation predictions"],
        [
            "Intended use",
            str(
                project_metadata.get(
                    "intended_use",
                    "Research and educational prototype only.",
                )
            ),
        ],
    ]

    project_table = Table(
        project_rows,
        colWidths=[48 * mm, 112 * mm],
        repeatRows=1,
    )
    project_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#e7f4f4"),
            ),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.extend([
        project_table,
        Spacer(1, 14),
        Paragraph("Evaluation Metrics", styles["Heading2"]),
    ])

    metric_rows = [["Metric", "Estimate"]]
    metric_rows.extend(
        [[name, f"{float(value):.4f}"] for name, value in metrics.items()]
    )

    metric_table = Table(
        metric_rows,
        colWidths=[80 * mm, 50 * mm],
        repeatRows=1,
    )
    metric_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#087f8c"),
            ),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.extend([
        metric_table,
        Spacer(1, 14),
        Paragraph("Limitations", styles["Heading2"]),
        Paragraph(
            "The results apply only to the dataset evaluated. They do "
            "not establish performance on independent Nigerian or "
            "African populations. Calibration does not establish "
            "clinical validity. The model must not be used for diagnosis, "
            "treatment, or patient triage.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Any claim of external validation requires documented "
            "dataset provenance, independence, suitable permissions, "
            "compatible feature definitions, and transparent reporting.",
            styles["BodyText"],
        ),
    ])

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            18 * mm,
            10 * mm,
            "AfriHealth AI | Research and education only",
        )
        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 4. HEADER AND RESEARCH DISCLAIMER
# ============================================================

st.markdown(
    '<div class="main-title">🫀 AfriHealth AI</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">AI-powered health-risk intelligence for Africa</div>',
    unsafe_allow_html=True,
)

st.write(
    "A research and educational prototype for exploring "
    "machine-learning-based cardiovascular classification."
)

st.warning(
    "RESEARCH AND EDUCATIONAL USE ONLY. This is not a diagnostic "
    "tool. The model uses the UCI Heart Disease dataset and has not "
    "been externally validated on Nigerian or broader African populations. "
    "Do not use its output for diagnosis, treatment, or clinical decisions."
)

st.divider()


# ============================================================
# 5. PROJECT OVERVIEW
# ============================================================

st.header("📊 Project Overview")

selected_model = metadata.get("selected_model", "Not recorded")
cv_auc = metadata.get(
    "cross_validation_roc_auc",
    metadata.get("cross_validation_auc"),
)

try:
    cv_auc_display = (
        f"{float(cv_auc):.4f}"
        if cv_auc is not None
        else "Not recorded"
    )
except (ValueError, TypeError):
    cv_auc_display = str(cv_auc) if cv_auc is not None else "Not recorded"

overview_cols = st.columns(4)

overview_cols[0].metric("Selected Model", str(selected_model))
overview_cols[1].metric("Cross-validation ROC-AUC", cv_auc_display)
overview_cols[2].metric(
    "Training Records",
    str(metadata.get("training_records", "N/A")),
)
overview_cols[3].metric(
    "Holdout Records",
    str(metadata.get("test_records", "N/A")),
)

st.caption(
    "Cross-validation ROC-AUC is a model-selection metric. It is not "
    "a personal disease probability or a guarantee of clinical performance."
)


# ============================================================
# 6. RESEARCH INPUT FORM AND PREDICTION
# ============================================================

st.divider()
st.header("🩺 Research Input")

st.write(
    "Enter values using the coding scheme expected by the saved model."
)

with st.form("risk_prediction_form"):
    st.subheader("Demographic Information")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            step=1,
        )

    with c2:
        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda value: (
                "Female (0)" if value == 0 else "Male (1)"
            ),
        )

    st.subheader("Clinical Measurements")
    c1, c2 = st.columns(2)

    with c1:
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

    with c2:
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda value: (
                "No (0)" if value == 0 else "Yes (1)"
            ),
        )

    st.subheader("Diagnostic Features")
    c1, c2 = st.columns(2)

    with c1:
        cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        exang = st.selectbox(
            "Exercise-Induced Angina (exang)",
            [0, 1],
            format_func=lambda value: (
                "No (0)" if value == 0 else "Yes (1)"
            ),
        )

    with c2:
        slope = st.selectbox("ST Segment Slope (slope)", [1, 2, 3])
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [3, 6, 7])

    submitted = st.form_submit_button(
        "🫀 Run AfriHealth AI Analysis",
        use_container_width=True,
    )

input_data = None
prediction = None
probability = None

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
        columns=FEATURES,
    )

    try:
        prediction = int(model.predict(input_data)[0])

        if prediction not in (0, 1):
            raise ValueError(
                "The model returned a class outside 0 and 1."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            positive_index = list(model.classes_).index(1)
            probability = float(probabilities[positive_index])

    except Exception as error:
        st.error("The model could not process these inputs.")
        st.caption(str(error))

    if prediction is not None:
        st.divider()
        st.header("📋 Analysis Result")

        if prediction == 1:
            st.warning("Model classification: Class 1")
        else:
            st.success("Model classification: Class 0")

        if probability is not None:
            st.metric(
                "Model-estimated probability for class 1",
                f"{probability * 100:.2f}%",
            )
        else:
            st.info("Probability output is not available.")

        st.warning(
            "This is a model classification, not a diagnosis. The "
            "probability is not a clinically validated personal risk "
            "estimate. Do not use it to make medical or treatment decisions."
        )

        with st.expander("View submitted research inputs"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# 7. LOAD HOLDOUT EVALUATION DATA
# ============================================================

evaluation_df = None
actual = predicted = probabilities = None
evaluation_error = None

if EVALUATION_PATH.is_file():
    try:
        evaluation_df = pd.read_csv(EVALUATION_PATH)
        actual, predicted, probabilities = validate_evaluation_data(
            evaluation_df
        )
    except Exception as error:
        evaluation_error = str(error)


# ============================================================
# 8. BUILD 04 - MODEL EVALUATION DASHBOARD
# ============================================================

st.divider()
st.header("📈 Model Evaluation Dashboard")

st.caption(
    "These metrics describe the exported holdout predictions only."
)

if evaluation_df is None:
    st.info(
        "Add evaluation_predictions.csv to the repository root "
        "to enable the evaluation dashboard."
    )
    if evaluation_error:
        st.error(evaluation_error)

else:
    metric_values = safe_metric(actual, predicted)
    metric_cols = st.columns(4)

    for column, (name, value) in zip(
        metric_cols,
        list(metric_values.items())[:4],
    ):
        column.metric(name, f"{value:.3f}")

    st.subheader("Confusion Matrix")
    st.plotly_chart(
        make_confusion_matrix_figure(actual, predicted),
        use_container_width=True,
    )

    st.caption(
        "The matrix shows correct and incorrect classifications "
        "in the exported holdout dataset."
    )

    if len(np.unique(actual)) == 2:
        st.subheader("Receiver Operating Characteristic (ROC) Curve")
        roc_fig, auc_value = make_roc_figure(
            actual,
            probabilities,
            title="Holdout ROC curve",
        )
        st.plotly_chart(roc_fig, use_container_width=True)
        st.metric("Holdout ROC-AUC", f"{auc_value:.3f}")

        metric_values["Holdout ROC-AUC"] = auc_value
    else:
        st.warning(
            "ROC-AUC requires both actual classes in the evaluation data."
        )

    st.subheader("Evaluation Summary")
    summary_df = pd.DataFrame({
        "Metric": list(metric_values.keys()),
        "Value": list(metric_values.values()),
    })

    st.dataframe(
        summary_df.style.format({"Value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Evaluation records: {len(actual)}. These results do not "
        "establish clinical validity or generalisation to African populations."
    )


# ============================================================
# 9. BUILD 05A - RELIABILITY AND VALIDATION
# ============================================================

st.divider()
st.header("🛡️ Model Reliability and Validation")

st.write(
    "This section examines classification performance and predicted "
    "probabilities on the available holdout sample."
)

if evaluation_df is None:
    st.info(
        "Reliability checks require a valid evaluation_predictions.csv."
    )
else:
    if len(np.unique(actual)) < 2:
        st.warning(
            "Both actual classes are required for the full reliability checks."
        )
    else:
        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        brier = brier_score_loss(actual, probabilities)
        balanced = balanced_accuracy_score(actual, predicted)

        reliability_cols = st.columns(4)
        reliability_cols[0].metric(
            "Balanced accuracy", f"{balanced:.3f}"
        )
        reliability_cols[1].metric(
            "Sensitivity", f"{sensitivity:.3f}"
        )
        reliability_cols[2].metric(
            "Specificity", f"{specificity:.3f}"
        )
        reliability_cols[3].metric(
            "Brier score", f"{brier:.3f}"
        )

        st.caption(
            "Sensitivity is the proportion of class-1 cases identified. "
            "Specificity is the proportion of class-0 cases identified. "
            "Lower Brier scores indicate better probability accuracy "
            "on this sample, but do not prove calibration."
        )

        st.subheader("Holdout Probability Calibration")
        try:
            calibration_fig = make_calibration_figure(
                actual,
                probabilities,
                bins=5,
                strategy="uniform",
                title="Holdout probability calibration",
            )
            st.plotly_chart(
                calibration_fig,
                use_container_width=True,
            )
        except Exception as error:
            st.warning("The holdout calibration chart could not be created.")
            st.caption(str(error))

        st.caption(
            "The curve compares average predicted probabilities with "
            "observed class-1 frequencies. A small holdout sample can "
            "produce unstable curves. This check does not establish "
            "clinical validity."
        )

        st.subheader("Classification Details")
        report = classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["Class 0", "Class 1"],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(
            report_df.style.format("{:.3f}"),
            use_container_width=True,
        )

        with st.expander("Validation Limitations"):
            st.markdown(
                """
                - The available holdout sample is limited.
                - Performance can change with a different sample.
                - Calibration on this dataset may not transfer to other
                  hospitals, countries, or populations.
                - These checks do not constitute independent external validation.
                - This prototype is not intended for diagnosis or treatment.
                """
            )


# ============================================================
# 10. BUILD 05B - PDF REPORT AND CSV DOWNLOADS
# ============================================================

st.divider()
st.header("📄 Downloadable Research Reports")

if evaluation_df is None:
    st.info(
        "A valid evaluation_predictions.csv is required to generate reports."
    )
else:
    try:
        report_metrics = safe_metric(actual, predicted)
        report_metrics["Brier score"] = brier_score_loss(
            actual, probabilities
        )
        report_metrics["Balanced accuracy"] = balanced_accuracy_score(
            actual, predicted
        )

        if len(np.unique(actual)) == 2:
            report_metrics["Holdout ROC-AUC"] = roc_auc_score(
                actual, probabilities
            )

        pdf_bytes = create_pdf_report(
            report_metrics,
            metadata,
            len(actual),
            "Model Evaluation and Reliability Report",
        )

        download_cols = st.columns(2)

        with download_cols[0]:
            st.download_button(
                "📄 Download PDF Research Report",
                data=pdf_bytes,
                file_name="AfriHealth_AI_Research_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with download_cols[1]:
            st.download_button(
                "📊 Download Evaluation Predictions CSV",
                data=evaluation_df.to_csv(index=False).encode("utf-8"),
                file_name="AfriHealth_AI_Evaluation_Predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.caption(
            "Review CSV contents and remove any sensitive information "
            "before sharing the file."
        )

    except Exception as error:
        st.error("The research report could not be generated.")
        st.caption(str(error))


# ============================================================
# 11. BUILD 04B - MODEL EXPLAINABILITY
# ============================================================

st.divider()
st.header("🔎 Model Explainability")

st.write(
    "Explore the learned coefficients of the Logistic Regression "
    "model and, after submitting an input, its feature contributions."
)

st.caption(
    "These explanations describe model behaviour, not medical causes "
    "or causal relationships."
)

try:
    if not hasattr(model, "steps") or len(model.steps) < 2:
        raise ValueError(
            "The loaded model is not a supported preprocessing pipeline."
        )

    estimator = model.steps[-1][1]
    preprocessing = model[:-1]

    if not hasattr(estimator, "coef_"):
        raise ValueError(
            "The final estimator does not expose Logistic Regression coefficients."
        )

    if not hasattr(preprocessing, "get_feature_names_out"):
        raise ValueError(
            "The preprocessing pipeline cannot provide transformed feature names."
        )

    transformed_names = preprocessing.get_feature_names_out()
    classes = list(estimator.classes_)
    coefficient_array = np.asarray(estimator.coef_)

    if len(classes) != 2 or coefficient_array.shape[0] != 1:
        raise ValueError(
            "Coefficient explanations currently support binary Logistic Regression."
        )

    if 1 not in classes:
        raise ValueError("Expected positive class 1 was not found.")

    # Binary Logistic Regression coefficients represent classes_[1].
    coef_class_1 = coefficient_array[0].copy()

    if classes.index(1) == 0:
        coef_class_1 = -coef_class_1

    if len(transformed_names) != len(coef_class_1):
        raise ValueError(
            "Transformed feature names do not match coefficient count."
        )

    importance_df = pd.DataFrame({
        "Feature": [
            str(name).replace("__", " / ")
            for name in transformed_names
        ],
        "Coefficient": coef_class_1,
        "Absolute coefficient": np.abs(coef_class_1),
    }).sort_values("Absolute coefficient", ascending=False)

    with st.expander("How to read these explanations"):
        st.markdown(
            """
            - **Global view:** ranks transformed features by absolute
              coefficient magnitude.
            - **Positive coefficient:** increasing the transformed feature
              pushes class-1 log-odds upward.
            - **Negative coefficient:** increasing the transformed feature
              pushes class-1 log-odds downward.
            - **Local view:** shows feature contributions for the submitted
              record relative to the model intercept.
            """
        )
        st.caption(
            "Categorical variables may be represented by multiple encoded "
            "columns. Coefficients are not causal effects."
        )

    st.subheader("1. Global Feature Influence")

    top_global = importance_df.head(15).sort_values(
        "Absolute coefficient", ascending=True
    )

    global_fig = go.Figure(
        go.Bar(
            x=top_global["Absolute coefficient"],
            y=top_global["Feature"],
            orientation="h",
            marker_color="#16c7b7",
            hovertemplate=(
                "Feature: %{y}<br>"
                "Absolute coefficient: %{x:.4f}<extra></extra>"
            ),
        )
    )

    global_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Absolute coefficient magnitude",
        yaxis_title="Transformed feature",
        height=520,
    )

    st.plotly_chart(global_fig, use_container_width=True)

    st.dataframe(
        importance_df.head(15)[
            ["Feature", "Coefficient", "Absolute coefficient"]
        ].style.format({
            "Coefficient": "{:.4f}",
            "Absolute coefficient": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Coefficient magnitude is not a measure of clinical importance. "
        "Preprocessing, encoding, and correlated features can affect ranking."
    )

    st.subheader("2. Individual Prediction Explanation")

    if submitted and input_data is not None and prediction is not None:
        transformed_input = preprocessing.transform(input_data)

        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()

        transformed_row = np.asarray(transformed_input)[0]
        contributions = transformed_row * coef_class_1

        intercept = float(
            np.asarray(estimator.intercept_).ravel()[0]
        )
        if classes.index(1) == 0:
            intercept = -intercept

        local_df = pd.DataFrame({
            "Feature": [
                str(name).replace("__", " / ")
                for name in transformed_names
            ],
            "Contribution": contributions,
        })

        local_df["Absolute contribution"] = np.abs(
            local_df["Contribution"]
        )

        local_df = local_df.sort_values(
            "Absolute contribution", ascending=False
        )

        top_local = local_df.head(12).sort_values("Contribution")
        top_local["Direction"] = np.where(
            top_local["Contribution"] >= 0,
            "Pushes towards class 1",
            "Pushes away from class 1",
        )

        local_fig = go.Figure(
            go.Bar(
                x=top_local["Contribution"],
                y=top_local["Feature"],
                orientation="h",
                marker_color=[
                    "#ff8c69" if value >= 0 else "#16c7b7"
                    for value in top_local["Contribution"]
                ],
                customdata=top_local[["Direction"]].to_numpy(),
                hovertemplate=(
                    "Feature: %{y}<br>"
                    "Contribution: %{x:.4f}<br>"
                    "%{customdata[0]}<extra></extra>"
                ),
            )
        )

        local_fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="#9fb0c0",
        )

        local_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Contribution to class-1 log-odds",
            yaxis_title="Transformed feature",
            height=480,
        )

        st.plotly_chart(local_fig, use_container_width=True)

        st.dataframe(
            local_df[
                ["Feature", "Contribution", "Absolute contribution"]
            ].style.format({
                "Contribution": "{:.4f}",
                "Absolute contribution": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.metric("Model intercept for class-1 log-odds", f"{intercept:.4f}")

        st.caption(
            "The intercept plus the sum of feature contributions reconstructs "
            "the class-1 log-odds for this binary Logistic Regression model. "
            "Contributions are not percentages or personal medical risk factors."
        )
    else:
        st.info(
            "Submit the Research Input form above to view an individual explanation."
        )

except Exception as error:
    st.warning(
        "Coefficient-based explainability is unavailable for the current model."
    )
    st.caption(f"Technical detail: {error}")


# ============================================================
# 12. MODEL INFORMATION
# ============================================================

st.divider()
st.header("🔬 Model Information")

with st.expander("View model and research details"):
    st.write("Selected model:", selected_model)
    st.write(
        "Selection metric:",
        metadata.get("selection_metric", "Not recorded"),
    )
    st.write("Cross-validation ROC-AUC:", cv_auc_display)
    st.write(
        "Dataset:",
        metadata.get("dataset", "UCI Heart Disease dataset"),
    )
    st.write(
        "Training records:",
        metadata.get("training_records", "Not recorded"),
    )
    st.write(
        "Holdout test records:",
        metadata.get("test_records", "Not recorded"),
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

with st.expander("📚 View input feature schema"):
    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Purpose": FEATURE_DESCRIPTIONS,
    })
    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

with st.expander("⚠️ Research limitations"):
    st.markdown(
        """
        **Dataset:** UCI Heart Disease dataset, repository ID 45.

        **Sample size:** The project records 303 samples, with 242 used
        for training and 61 reserved for holdout testing, according to
        the current project description.

        **Model selection:** Logistic Regression was selected using
        cross-validation ROC-AUC.

        **Population limitation:** The source dataset is not representative
        of the Nigerian or broader African population.

        **External validation:** Independent validation on Nigerian or
        African clinical data has not been established by this app.

        **Clinical use:** Do not use this prototype for diagnosis,
        treatment, patient triage, or clinical decisions.

        **Future research:** Representative datasets, external validation,
        calibration, fairness assessment, privacy safeguards, and clinical
        collaboration are needed.
        """
    )


# ============================================================
# 13. BUILD 06A / 06B - EXTERNAL VALIDATION WORKSPACE
# ============================================================

st.divider()
st.header("🧪 Build 06: External Validation Workspace")

st.write(
    "Evaluate the frozen saved model on a separately sourced dataset. "
    "This workflow does not train or retrain the model."
)

st.caption(
    "Required columns: all 13 model features and a binary target column "
    "named target. The target must already use 0 and 1."
)

st.code(", ".join(FEATURES + ["target"]))

st.warning(
    "Upload only authorised, de-identified data. Do not upload names, "
    "addresses, phone numbers, medical record numbers, or other "
    "directly identifying patient information."
)

external_file = st.file_uploader(
    "Upload an authorised validation CSV",
    type=["csv"],
    key="external_validation_csv",
)

external_source = st.text_input(
    "Dataset source or study identifier",
    placeholder="Example: independent cohort, study identifier",
    key="external_validation_source",
)

if external_file is not None:
    st.caption(
        "Check dataset provenance, permissions, target definition, feature "
        "coding, and independence from model development before interpreting results."
    )

    if st.button(
        "Run validation on uploaded dataset",
        key="run_external_validation",
        type="primary",
        use_container_width=True,
    ):
        try:
            external_df = pd.read_csv(external_file)

            required_external = set(FEATURES + ["target"])
            missing_external = required_external - set(external_df.columns)

            if missing_external:
                st.error(
                    "Missing required columns: "
                    + ", ".join(sorted(missing_external))
                )

            elif external_df.empty:
                st.error("The uploaded dataset contains no records.")

            else:
                X_external = external_df[FEATURES].copy()
                X_external = X_external.apply(
                    pd.to_numeric, errors="coerce"
                )

                y_series = pd.to_numeric(
                    external_df["target"], errors="coerce"
                )

                if X_external.isnull().any().any():
                    bad_columns = X_external.columns[
                        X_external.isnull().any()
                    ].tolist()
                    st.error(
                        "Missing or non-numeric feature values in: "
                        + ", ".join(bad_columns)
                    )

                elif y_series.isnull().any():
                    st.error(
                        "The target column contains missing or "
                        "non-numeric values."
                    )

                elif not np.isfinite(
                    X_external.to_numpy(dtype=float)
                ).all():
                    st.error("All feature values must be finite numbers.")

                elif not np.isin(y_series.to_numpy(), [0, 1]).all():
                    st.error(
                        "The target must already be binary: 0 or 1. "
                        "Apply the documented target transformation before upload."
                    )

                elif y_series.nunique() < 2:
                    st.error(
                        "The dataset must contain both target classes."
                    )

                else:
                    y_external = y_series.to_numpy(dtype=int)

                    if not hasattr(model, "predict_proba"):
                        st.error(
                            "The saved model does not support probability predictions."
                        )

                    else:
                        positive_index = list(model.classes_).index(1)

                        external_predictions = np.asarray(
                            model.predict(X_external)
                        ).astype(int)

                        external_probabilities = np.asarray(
                            model.predict_proba(X_external)[:, positive_index],
                            dtype=float,
                        )

                        if not np.isin(
                            external_predictions, [0, 1]
                        ).all():
                            st.error(
                                "The model returned predictions outside classes 0 and 1."
                            )

                        elif (
                            not np.isfinite(
                                external_probabilities
                            ).all()
                            or (
                                (external_probabilities < 0)
                                | (external_probabilities > 1)
                            ).any()
                        ):
                            st.error(
                                "The model produced invalid probabilities."
                            )

                        else:
                            tn, fp, fn, tp = confusion_matrix(
                                y_external,
                                external_predictions,
                                labels=[0, 1],
                            ).ravel()

                            sensitivity = (
                                tp / (tp + fn) if tp + fn else 0.0
                            )
                            specificity = (
                                tn / (tn + fp) if tn + fp else 0.0
                            )

                            external_metrics = {
                                "Accuracy": accuracy_score(
                                    y_external, external_predictions
                                ),
                                "Balanced accuracy": balanced_accuracy_score(
                                    y_external, external_predictions
                                ),
                                "Precision": precision_score(
                                    y_external,
                                    external_predictions,
                                    zero_division=0,
                                ),
                                "Sensitivity": sensitivity,
                                "Specificity": specificity,
                                "F1 score": f1_score(
                                    y_external,
                                    external_predictions,
                                    zero_division=0,
                                ),
                                "ROC-AUC": roc_auc_score(
                                    y_external,
                                    external_probabilities,
                                ),
                                "Brier score": brier_score_loss(
                                    y_external,
                                    external_probabilities,
                                ),
                            }

                            # Stratified bootstrap:
                            # sample each actual class separately so both
                            # classes remain represented in every replicate.
                            rng = np.random.default_rng(2026)
                            indices_0 = np.flatnonzero(y_external == 0)
                            indices_1 = np.flatnonzero(y_external == 1)

                            bootstrap_results = {
                                name: [] for name in external_metrics
                            }

                            for _ in range(1000):
                                sampled_indices = np.concatenate([
                                    rng.choice(
                                        indices_0,
                                        size=len(indices_0),
                                        replace=True,
                                    ),
                                    rng.choice(
                                        indices_1,
                                        size=len(indices_1),
                                        replace=True,
                                    ),
                                ])

                                y_boot = y_external[sampled_indices]
                                pred_boot = external_predictions[sampled_indices]
                                prob_boot = external_probabilities[sampled_indices]

                                tn_b, fp_b, fn_b, tp_b = confusion_matrix(
                                    y_boot, pred_boot, labels=[0, 1]
                                ).ravel()

                                boot_sensitivity = (
                                    tp_b / (tp_b + fn_b)
                                    if tp_b + fn_b else 0.0
                                )
                                boot_specificity = (
                                    tn_b / (tn_b + fp_b)
                                    if tn_b + fp_b else 0.0
                                )

                                boot_metrics = {
                                    "Accuracy": accuracy_score(
                                        y_boot, pred_boot
                                    ),
                                    "Balanced accuracy": balanced_accuracy_score(
                                        y_boot, pred_boot
                                    ),
                                    "Precision": precision_score(
                                        y_boot,
                                        pred_boot,
                                        zero_division=0,
                                    ),
                                    "Sensitivity": boot_sensitivity,
                                    "Specificity": boot_specificity,
                                    "F1 score": f1_score(
                                        y_boot,
                                        pred_boot,
                                        zero_division=0,
                                    ),
                                    "ROC-AUC": roc_auc_score(
                                        y_boot, prob_boot
                                    ),
                                    "Brier score": brier_score_loss(
                                        y_boot, prob_boot
                                    ),
                                }

                                for name, value in boot_metrics.items():
                                    bootstrap_results[name].append(value)

                            ci_rows = []

                            for name, estimate in external_metrics.items():
                                lower, upper = np.percentile(
                                    bootstrap_results[name],
                                    [2.5, 97.5],
                                )

                                ci_rows.append({
                                    "Metric": name,
                                    "Estimate": estimate,
                                    "95% CI lower": lower,
                                    "95% CI upper": upper,
                                })

                            ci_df = pd.DataFrame(ci_rows)

                            st.subheader("External Validation Results")
                            st.write(
                                "Dataset source:",
                                external_source.strip() or "Not supplied",
                            )
                            st.write(
                                f"Records evaluated: {len(y_external)}"
                            )
                            st.write(
                                f"Class 0: {len(indices_0)} | "
                                f"Class 1: {len(indices_1)}"
                            )

                            st.dataframe(
                                ci_df.style.format({
                                    "Estimate": "{:.3f}",
                                    "95% CI lower": "{:.3f}",
                                    "95% CI upper": "{:.3f}",
                                }),
                                use_container_width=True,
                                hide_index=True,
                            )

                            st.caption(
                                "Intervals are percentile intervals from "
                                "1,000 class-stratified bootstrap replicates. "
                                "They reflect sampling variation under this "
                                "procedure, not all sources of uncertainty."
                            )

                            st.subheader("External Dataset ROC Curve")
                            roc_fig, _ = make_roc_figure(
                                y_external,
                                external_probabilities,
                                title="External dataset ROC curve",
                            )
                            st.plotly_chart(
                                roc_fig,
                                use_container_width=True,
                            )

                            # BUILD 06B: calibration appears only after
                            # external predictions and labels are available.
                            st.subheader("External Probability Calibration")

                            calibration_fig = make_calibration_figure(
                                y_external,
                                external_probabilities,
                                bins=5,
                                strategy="quantile",
                                title="External probability calibration",
                            )

                            st.plotly_chart(
                                calibration_fig,
                                use_container_width=True,
                            )

                            st.caption(
                                "This chart compares predicted probabilities "
                                "with observed class-1 frequencies in the "
                                "uploaded dataset. Small datasets may produce "
                                "unstable curves. Calibration does not prove "
                                "clinical validity."
                            )

                            st.subheader("Download Validation Results")

                            results_df = X_external.copy()
                            results_df["y_true"] = y_external
                            results_df["y_pred"] = external_predictions
                            results_df["prob_class_1"] = external_probabilities

                            download_cols = st.columns(2)

                            with download_cols[0]:
                                st.download_button(
                                    "Download predictions CSV",
                                    data=results_df.to_csv(
                                        index=False
                                    ).encode("utf-8"),
                                    file_name=(
                                        "AfriHealth_Validation_Predictions.csv"
                                    ),
                                    mime="text/csv",
                                    key="download_external_predictions",
                                    use_container_width=True,
                                )

                            with download_cols[1]:
                                st.download_button(
                                    "Download metrics and confidence intervals",
                                    data=ci_df.to_csv(
                                        index=False
                                    ).encode("utf-8"),
                                    file_name=(
                                        "AfriHealth_Validation_Metrics.csv"
                                    ),
                                    mime="text/csv",
                                    key="download_external_metrics",
                                    use_container_width=True,
                                )

                            st.warning(
                                "Call this external validation only after "
                                "confirming dataset independence, provenance, "
                                "appropriate permissions, and compatible feature "
                                "definitions. Results do not establish clinical "
                                "validity or suitability for medical decisions."
                            )

        except Exception as error:
            st.error(
                "Validation could not be completed. Check the CSV schema, "
                "feature coding, target labels, and model compatibility."
            )
            st.caption(str(error))


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "AfriHealth AI | AI-powered health-risk intelligence for Africa | "
    "Research and educational prototype | Version 0.1.0"
)
