
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)


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
# BUILD 04 - MODEL EVALUATION DASHBOARD
# ============================================================

EVALUATION_PATH = BASE_DIR / "evaluation_predictions.csv"

st.markdown(
    '<div class="section-title">Model Evaluation Dashboard</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Metrics below are calculated from the exported holdout "
    "predictions. They describe performance on this dataset only."
)

if not EVALUATION_PATH.exists():

    st.info(
        "Evaluation data is not available yet. Add "
        "evaluation_predictions.csv to the repository root "
        "alongside app.py."
    )

else:

    try:
        evaluation_df = pd.read_csv(EVALUATION_PATH)

        required_columns = {
            "y_true",
            "y_pred",
            "prob_class_1",
        }

        if not required_columns.issubset(evaluation_df.columns):
            st.error(
                "The evaluation CSV is missing required columns: "
                + ", ".join(
                    sorted(required_columns - set(evaluation_df.columns))
                )
            )
            st.stop()

        if evaluation_df.empty:
            st.error("The evaluation CSV contains no rows.")
            st.stop()

        if evaluation_df[list(required_columns)].isnull().any().any():
            st.error(
                "The evaluation CSV contains missing values "
                "in required columns."
            )
            st.stop()

        y_true_eval = evaluation_df["y_true"].astype(int).to_numpy()
        y_pred_eval = evaluation_df["y_pred"].astype(int).to_numpy()
        prob_eval = evaluation_df["prob_class_1"].astype(float).to_numpy()

        if not set(np.unique(y_true_eval)).issubset({0, 1}):
            st.error("Actual labels must contain only 0 and 1.")
            st.stop()

        if not set(np.unique(y_pred_eval)).issubset({0, 1}):
            st.error("Predicted labels must contain only 0 and 1.")
            st.stop()

        if not np.isfinite(prob_eval).all():
            st.error("The probability column contains invalid values.")
            st.stop()

        if ((prob_eval < 0) | (prob_eval > 1)).any():
            st.error("Probabilities must be between 0 and 1.")
            st.stop()

        if len(np.unique(y_true_eval)) < 2:
            st.warning(
                "ROC-AUC and the ROC curve require both actual "
                "classes to be present in the evaluation data."
            )

        # Calculate metrics from the actual exported predictions.
        accuracy = accuracy_score(y_true_eval, y_pred_eval)
        precision = precision_score(
            y_true_eval, y_pred_eval, zero_division=0
        )
        recall = recall_score(
            y_true_eval, y_pred_eval, zero_division=0
        )
        f1 = f1_score(
            y_true_eval, y_pred_eval, zero_division=0
        )

        metric_cols = st.columns(4)

        metric_values = [
            ("Accuracy", accuracy),
            ("Precision", precision),
            ("Recall", recall),
            ("F1 Score", f1),
        ]

        for column, (label, value) in zip(metric_cols, metric_values):
            with column:
                st.metric(label, f"{value:.3f}")

        # Confusion matrix
        st.markdown("### Confusion Matrix")

        cm = confusion_matrix(
            y_true_eval,
            y_pred_eval,
            labels=[0, 1],
        )

        cm_figure = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                colorscale=[
                    [0, "#102235"],
                    [0.5, "#087f8c"],
                    [1, "#5fffe9"],
                ],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 18, "color": "white"},
                showscale=False,
            )
        )

        cm_figure.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
            height=380,
            margin=dict(l=30, r=20, t=20, b=40),
        )

        st.plotly_chart(
            cm_figure,
            use_container_width=True,
        )

        st.caption(
            "Class 0 and class 1 follow the project's original "
            "binary target transformation. The matrix counts "
            "correct and incorrect classifications in the holdout set."
        )

        # ROC curve
        st.markdown("### Receiver Operating Characteristic (ROC) Curve")

        if len(np.unique(y_true_eval)) == 2:

            fpr, tpr, thresholds = roc_curve(
                y_true_eval,
                prob_eval,
                pos_label=1,
            )

            auc_value = roc_auc_score(y_true_eval, prob_eval)

            roc_figure = go.Figure()

            roc_figure.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"Holdout ROC (AUC = {auc_value:.3f})",
                    line=dict(color="#5fffe9", width=3),
                )
            )

            roc_figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Chance reference",
                    line=dict(
                        color="#9fb0c0",
                        width=2,
                        dash="dash",
                    ),
                )
            )

            roc_figure.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.02]),
                height=420,
                legend=dict(orientation="h", y=-0.2),
                margin=dict(l=30, r=20, t=25, b=70),
            )

            st.plotly_chart(
                roc_figure,
                use_container_width=True,
            )

            st.metric(
                "Holdout ROC-AUC",
                f"{auc_value:.3f}",
            )

            st.caption(
                "This holdout ROC-AUC is calculated from the exported "
                "class-1 probabilities. It is distinct from the mean "
                "cross-validation ROC-AUC shown in the project metadata."
            )

        # Evaluation table
        st.markdown("### Evaluation Summary")

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                ],
                "Value": [
                    accuracy,
                    precision,
                    recall,
                    f1,
                ],
            }
        )

        if len(np.unique(y_true_eval)) == 2:
            summary_df.loc[len(summary_df)] = [
                "Holdout ROC-AUC",
                roc_auc_score(y_true_eval, prob_eval),
            ]

        st.dataframe(
            summary_df.style.format({"Value": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Evaluation rows: {len(evaluation_df)}. "
            "These results do not establish clinical validity, "
            "generalisation to Nigeria, or suitability for medical use."
        )

    except Exception as error:
        st.error("The evaluation dashboard could not load the CSV.")
        st.exception(error)


# ============================================================
# BUILD 05A - MODEL RELIABILITY AND VALIDATION
# ============================================================

from sklearn.metrics import (
    brier_score_loss,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

st.divider()
st.header("🛡️ Model Reliability and Validation")

st.write(
    "These checks describe the model's performance on the "
    "available holdout dataset. They do not establish clinical "
    "validity or performance on Nigerian or other African populations."
)

reliability_path = BASE_DIR / "evaluation_predictions.csv"

if not reliability_path.exists():
    st.warning(
        "Add evaluation_predictions.csv to the repository root "
        "to enable reliability checks."
    )
else:
    try:
        reliability_df = pd.read_csv(reliability_path)

        needed = {"y_true", "y_pred", "prob_class_1"}
        if not needed.issubset(reliability_df.columns):
            raise ValueError(
                "The CSV must contain y_true, y_pred, and prob_class_1."
            )

        actual = pd.to_numeric(
            reliability_df["y_true"], errors="raise"
        ).to_numpy(dtype=int)

        predicted = pd.to_numeric(
            reliability_df["y_pred"], errors="raise"
        ).to_numpy(dtype=int)

        probability = pd.to_numeric(
            reliability_df["prob_class_1"], errors="raise"
        ).to_numpy(dtype=float)

        if len(actual) == 0:
            raise ValueError("The evaluation dataset is empty.")

        if not (
            len(actual) == len(predicted) == len(probability)
        ):
            raise ValueError("Evaluation columns have different lengths.")

        if not np.isin(actual, [0, 1]).all():
            raise ValueError("Actual labels must be 0 or 1.")

        if not np.isin(predicted, [0, 1]).all():
            raise ValueError("Predicted labels must be 0 or 1.")

        if not np.isfinite(probability).all():
            raise ValueError("Probabilities must be finite numbers.")

        if ((probability < 0) | (probability > 1)).any():
            raise ValueError("Probabilities must be between 0 and 1.")

        if len(np.unique(actual)) < 2:
            st.warning(
                "Both actual classes are required for the full set "
                "of reliability checks."
            )
        else:
            tn, fp, fn, tp = confusion_matrix(
                actual, predicted, labels=[0, 1]
            ).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0

            reliability_cols = st.columns(4)

            reliability_cols[0].metric(
                "Balanced accuracy",
                f"{balanced_accuracy_score(actual, predicted):.3f}",
            )
            reliability_cols[1].metric(
                "Sensitivity",
                f"{sensitivity:.3f}",
            )
            reliability_cols[2].metric(
                "Specificity",
                f"{specificity:.3f}",
            )
            reliability_cols[3].metric(
                "Brier score",
                f"{brier_score_loss(actual, probability):.3f}",
            )

            st.caption(
                "Sensitivity measures the fraction of class-1 cases "
                "identified. Specificity measures the fraction of "
                "class-0 cases identified. Lower Brier scores indicate "
                "better probability accuracy on this evaluation sample, "
                "but do not prove that probabilities are calibrated."
            )

            st.subheader("Probability Calibration Check")

            observed_rate, mean_probability = calibration_curve(
                actual,
                probability,
                n_bins=5,
                strategy="uniform",
                pos_label=1,
            )

            calibration_fig = go.Figure()

            calibration_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Ideal calibration",
                    line=dict(dash="dash", color="#9fb0c0"),
                )
            )

            calibration_fig.add_trace(
                go.Scatter(
                    x=mean_probability,
                    y=observed_rate,
                    mode="lines+markers",
                    name="Observed holdout calibration",
                    line=dict(color="#5fffe9", width=3),
                    marker=dict(size=9),
                )
            )

            calibration_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Mean predicted class-1 probability",
                yaxis_title="Observed class-1 fraction",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=420,
            )

            st.plotly_chart(
                calibration_fig,
                use_container_width=True,
            )

            st.caption(
                "Each point compares average predicted probability "
                "with the observed class-1 fraction in a probability "
                "bin. With a small holdout sample, the curve can be "
                "unstable. This is a diagnostic, not proof of calibration."
            )

            st.subheader("Classification Details")

            report = classification_report(
                actual,
                predicted,
                labels=[0, 1],
                target_names=[
                    "Class 0",
                    "Class 1",
                ],
                output_dict=True,
                zero_division=0,
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.style.format("{:.3f}"),
                use_container_width=True,
            )

            st.subheader("Validation Limitations")

            st.markdown(
                """
                - The evaluation uses a limited holdout sample from
                  the UCI Heart Disease dataset.
                - Results can change with a different sample or split.
                - Calibration on this dataset may not transfer to other
                  hospitals, countries, or populations.
                - No independent Nigerian or African validation has
                  been established by these checks.
                - The output is not a diagnosis and should not be used
                  to make treatment decisions.
                """
            )

            st.success(
                f"Reliability checks completed for {len(actual)} "
                "holdout records."
            )

    except Exception as error:
        st.error("Reliability checks could not be completed.")
        st.caption(str(error))


# ============================================================
# BUILD 05B - DOWNLOADABLE RESEARCH REPORTS
# ============================================================

from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

st.divider()
st.header("📄 Downloadable Research Reports")

report_eval_path = BASE_DIR / "evaluation_predictions.csv"
report_metadata_path = BASE_DIR / "model_metadata.json"


def create_afrihealth_pdf(metrics, metadata, record_count):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AfriHealth AI Research Evaluation Report",
        author="AfriHealth AI",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#087f8c"),
            spaceAfter=12,
        )
    )

    story = [
        Paragraph(
            "AfriHealth AI",
            styles["ReportTitle"],
        ),
        Paragraph(
            "Model Evaluation and Reliability Report",
            styles["Heading2"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Purpose: research and education only. This report "
            "does not establish clinical validity and is not intended "
            "for diagnosis or treatment decisions.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Project Information", styles["Heading2"]),
    ]

    project_rows = [
        ["Project", "AfriHealth AI"],
        ["Model", str(metadata.get("selected_model", "Not specified"))],
        ["Dataset", str(metadata.get(
            "dataset", "UCI Heart Disease dataset"
        ))],
        ["Evaluation records", str(record_count)],
        ["Evaluation basis", "Exported holdout predictions"],
        ["Project version", str(metadata.get("version", "Not specified"))],
        ["Intended use", str(metadata.get(
            "intended_use",
            "Research and educational screening-support prototype only."
        ))],
    ]

    project_table = Table(
        project_rows,
        colWidths=[48 * mm, 112 * mm],
        repeatRows=1,
    )
    project_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f4f4")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.extend([
        project_table,
        Spacer(1, 14),
        Paragraph("Holdout Evaluation Metrics", styles["Heading2"]),
    ])

    metric_rows = [["Metric", "Value"]]
    metric_rows.extend(
        [[name, f"{value:.4f}"] for name, value in metrics.items()]
    )

    metric_table = Table(
        metric_rows,
        colWidths=[80 * mm, 50 * mm],
        repeatRows=1,
    )
    metric_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087f8c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.extend([
        metric_table,
        Spacer(1, 14),
        Paragraph("Interpretation and Limitations", styles["Heading2"]),
        Paragraph(
            "These metrics describe performance on the exported "
            "holdout sample only. They do not establish performance "
            "on independent data or in Nigerian or wider African "
            "populations. Probability estimates have not been shown "
            "to be clinically calibrated. A model output must not "
            "be interpreted as a diagnosis or a treatment recommendation.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Research integrity: report the dataset, split procedure, "
            "model-selection method, and limitations when presenting "
            "these findings. Do not claim external validation unless "
            "it has actually been performed.",
            styles["BodyText"],
        ),
    ])

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            18 * mm, 10 * mm,
            "AfriHealth AI | Research and education only",
        )
        canvas.drawRightString(
            A4[0] - 18 * mm, 10 * mm,
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


if report_eval_path.exists():
    try:
        report_df = pd.read_csv(report_eval_path)
        report_required = {"y_true", "y_pred", "prob_class_1"}

        if not report_required.issubset(report_df.columns):
            raise ValueError(
                "Evaluation CSV is missing required columns."
            )

        report_actual = pd.to_numeric(
            report_df["y_true"], errors="raise"
        ).to_numpy(dtype=int)
        report_predicted = pd.to_numeric(
            report_df["y_pred"], errors="raise"
        ).to_numpy(dtype=int)
        report_probability = pd.to_numeric(
            report_df["prob_class_1"], errors="raise"
        ).to_numpy(dtype=float)

        if len(report_actual) == 0:
            raise ValueError("Evaluation CSV contains no records.")

        if not (
            len(report_actual)
            == len(report_predicted)
            == len(report_probability)
        ):
            raise ValueError("Evaluation columns have different lengths.")

        if not np.isin(report_actual, [0, 1]).all():
            raise ValueError("Actual labels must be 0 or 1.")

        if not np.isin(report_predicted, [0, 1]).all():
            raise ValueError("Predicted labels must be 0 or 1.")

        if (
            not np.isfinite(report_probability).all()
            or ((report_probability < 0) | (report_probability > 1)).any()
        ):
            raise ValueError("Invalid probability values found.")

        report_metrics = {
            "Accuracy": accuracy_score(
                report_actual, report_predicted
            ),
            "Precision": precision_score(
                report_actual, report_predicted, zero_division=0
            ),
            "Recall / Sensitivity": recall_score(
                report_actual, report_predicted, zero_division=0
            ),
            "F1 score": f1_score(
                report_actual, report_predicted, zero_division=0
            ),
            "Balanced accuracy": balanced_accuracy_score(
                report_actual, report_predicted
            ),
            "Brier score": brier_score_loss(
                report_actual, report_probability
            ),
        }

        if len(np.unique(report_actual)) == 2:
            report_metrics["Holdout ROC-AUC"] = roc_auc_score(
                report_actual, report_probability
            )

        if report_metadata_path.exists():
            with open(report_metadata_path, "r", encoding="utf-8") as file:
                report_metadata = json.load(file)
        else:
            report_metadata = {
                "project": "AfriHealth AI",
                "version": "Not specified",
                "selected_model": "Logistic Regression",
                "dataset": "UCI Heart Disease dataset, repository ID 45",
                "intended_use": (
                    "Research and educational screening-support prototype only; "
                    "not for diagnosis or treatment decisions."
                ),
            }

        pdf_bytes = create_afrihealth_pdf(
            report_metrics,
            report_metadata,
            len(report_actual),
        )

        st.download_button(
            label="Download PDF Research Report",
            data=pdf_bytes,
            file_name="AfriHealth_AI_Research_Report.pdf",
            mime="application/pdf",
        )

        st.download_button(
            label="Download Evaluation Predictions CSV",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="AfriHealth_AI_Evaluation_Predictions.csv",
            mime="text/csv",
        )

        st.caption(
            "The CSV contains evaluation labels and model predictions. "
            "Review its contents before sharing it publicly."
        )

    except Exception as error:
        st.error("The research report could not be generated.")
        st.caption(str(error))
else:
    st.info(
        "Upload evaluation_predictions.csv to enable report downloads."
    )
    
# ============================================================
# BUILD 04B - MODEL EXPLAINABILITY
# ============================================================

st.divider()
st.header("🔎 Model Explainability")

st.write(
    "Explore which transformed input features influence the "
    "Logistic Regression model and how they contribute to an "
    "individual model output."
)

st.caption(
    "These explanations describe model behaviour, not medical "
    "causes. Coefficient magnitudes depend on preprocessing and "
    "feature encoding."
)

try:
    # Extract the final estimator and preprocessing pipeline.
    if not hasattr(model, "steps") or len(model.steps) < 2:
        raise ValueError(
            "The loaded model is not a supported preprocessing pipeline."
        )

    estimator = model.steps[-1][1]
    preprocessing = model[:-1]

    if not hasattr(estimator, "coef_"):
        raise ValueError(
            "This model does not expose coefficients. "
            "Coefficient-based explanations require a compatible estimator."
        )

    if not hasattr(preprocessing, "get_feature_names_out"):
        raise ValueError(
            "The preprocessing pipeline cannot provide transformed feature names."
        )

    transformed_names = preprocessing.get_feature_names_out()
    classes = list(estimator.classes_)

    if 1 not in classes:
        raise ValueError(
            "The model does not contain the expected positive class 1."
        )

    # For binary Logistic Regression, coef_ represents classes_[1].
    coefficients = np.asarray(estimator.coef_)

    if coefficients.shape[0] != 1 or len(classes) != 2:
        raise ValueError(
            "This explanation block currently supports binary Logistic Regression."
        )

    class_1_index = classes.index(1)
    coef_class_1 = (
        coefficients[0]
        if class_1_index == 1
        else -coefficients[0]
    )

    if len(transformed_names) != len(coef_class_1):
        raise ValueError(
            "Feature names and model coefficients have different lengths."
        )

    importance_df = pd.DataFrame({
        "Feature": [
            str(name).replace("__", " / ")
            for name in transformed_names
        ],
        "Coefficient": coef_class_1,
        "Absolute coefficient": np.abs(coef_class_1),
    })

    importance_df = importance_df.sort_values(
        "Absolute coefficient",
        ascending=False,
    )

    with st.expander("How to read these explanations"):
        st.markdown(
            """
            - **Global view:** ranks transformed features by the
              absolute size of their learned coefficients.
            - **Positive coefficient:** pushes the model's class-1
              log-odds upward when the transformed feature increases.
            - **Negative coefficient:** pushes class-1 log-odds downward
              when the transformed feature increases.
            - **Local view:** shows the signed contributions for the
              submitted record, relative to the model's intercept.
            """
        )
        st.caption(
            "Categorical variables may be represented by multiple "
            "encoded columns. Coefficients are not causal effects."
        )

    # GLOBAL EXPLANATION
    st.subheader("1. Global Feature Influence")

    st.write(
        "These are the 15 transformed features with the largest "
        "absolute Logistic Regression coefficients."
    )

    top_global = importance_df.head(15).sort_values(
        "Absolute coefficient",
        ascending=True,
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
        margin=dict(l=20, r=20, t=20, b=40),
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
        "This ranks coefficient magnitude, not clinical importance "
        "or the strength of a causal relationship. Correlated features "
        "and encoding choices can affect the ranking."
    )

    # INDIVIDUAL EXPLANATION
    st.subheader("2. Individual Prediction Explanation")

    if submitted and "input_data" in locals():

        transformed_input = preprocessing.transform(input_data)

        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()

        transformed_row = np.asarray(transformed_input)[0]

        contributions = transformed_row * coef_class_1

        intercept = float(np.asarray(estimator.intercept_).ravel()[0])

        if class_1_index == 0:
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
            "Absolute contribution",
            ascending=False,
        )

        st.write(
            "The chart shows which transformed input features "
            "contributed most to the model's class-1 log-odds "
            "for the submitted record."
        )

        top_local = local_df.head(12).copy()
        top_local = top_local.sort_values("Contribution")

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
            margin=dict(l=20, r=20, t=20, b=40),
        )

        st.plotly_chart(local_fig, use_container_width=True)

        st.caption(
            "Positive contributions increase class-1 log-odds; "
            "negative contributions decrease them. These values are "
            "not percentage points or individual medical risk factors."
        )

        with st.expander("View contribution details"):
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

        st.metric(
            "Model intercept for class-1 log-odds",
            f"{intercept:.4f}",
        )

        st.caption(
            "For binary Logistic Regression, the intercept plus all "
            "feature contributions reconstructs the class-1 log-odds "
            "before applying the logistic function. Contributions depend "
            "on the pipeline's scaling and encoding."
        )

    else:
        st.info(
            "Submit the Research Input form above to view an "
            "individual prediction explanation."
        )

except Exception as error:
    st.warning(
        "The explainability section could not be generated for this "
        "model pipeline. Your prediction and evaluation sections "
        "can still be used."
    )
    st.caption(f"Technical detail: {error}")
    
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
