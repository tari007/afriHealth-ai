from pathlib import Path

app_code = r'''"""
AfriHealth AI
Research and educational cardiovascular classification prototype.
Not for diagnosis, treatment, triage, or clinical decision-making.
"""

import json
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

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

# ============================================================
# PAGE CONFIGURATION AND PATHS
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

DEFAULT_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal",
]

# The model was trained against this feature order and coding.
EXTERNAL_FEATURES = DEFAULT_FEATURES.copy()


# ============================================================
# SHARED HELPERS
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH.name}")
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_binary_evaluation_frame(frame, required_columns):
    """Validate labels and probabilities in a saved evaluation CSV."""
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("The CSV contains no records.")

    actual = pd.to_numeric(frame["y_true"], errors="raise").to_numpy(dtype=int)
    predicted = pd.to_numeric(frame["y_pred"], errors="raise").to_numpy(dtype=int)
    probability = pd.to_numeric(
        frame["prob_class_1"], errors="raise"
    ).to_numpy(dtype=float)

    if not (len(actual) == len(predicted) == len(probability)):
        raise ValueError("Evaluation columns have different lengths.")
    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Actual labels must be 0 or 1.")
    if not np.isin(predicted, [0, 1]).all():
        raise ValueError("Predicted labels must be 0 or 1.")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite numbers.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    return actual, predicted, probability


def binary_metrics(actual, predicted, probability):
    """Return commonly used binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    result = {
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced accuracy": balanced_accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 score": f1_score(actual, predicted, zero_division=0),
        "Brier score": brier_score_loss(actual, probability),
    }
    if len(np.unique(actual)) == 2:
        result["ROC-AUC"] = roc_auc_score(actual, probability)
    return result


def make_roc_figure(actual, probability, title_prefix=""):
    fpr, tpr, _ = roc_curve(actual, probability, pos_label=1)
    auc_value = roc_auc_score(actual, probability)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC-AUC = {auc_value:.3f}",
        line=dict(color="#5fffe9", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Chance reference",
        line=dict(color="#9fb0c0", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=title_prefix or None,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        height=420,
        margin=dict(l=30, r=20, t=35, b=40),
    )
    return fig, auc_value


def make_calibration_figure(actual, probability, bins=5, strategy="uniform"):
    observed, mean_probability = calibration_curve(
        actual,
        probability,
        n_bins=bins,
        strategy=strategy,
        pos_label=1,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Ideal calibration",
        line=dict(dash="dash", color="#9fb0c0"),
    ))
    fig.add_trace(go.Scatter(
        x=mean_probability, y=observed, mode="lines+markers",
        name="Observed calibration",
        line=dict(color="#5fffe9", width=3),
        marker=dict(size=8),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Mean predicted class-1 probability",
        yaxis_title="Observed class-1 fraction",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=420,
        margin=dict(l=30, r=20, t=25, b=40),
    )
    return fig


def create_afrihealth_pdf(metrics, metadata, record_count, evaluation_basis):
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
    styles.add(ParagraphStyle(
        name="AfriHealthTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#087f8c"),
        spaceAfter=12,
    ))

    story = [
        Paragraph("AfriHealth AI", styles["AfriHealthTitle"]),
        Paragraph("Model Evaluation and Reliability Report", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph(
            "Research and educational use only. This report does not establish "
            "clinical validity and must not be used for diagnosis, treatment, "
            "or patient-triage decisions.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Project Information", styles["Heading2"]),
    ]

    project_rows = [
        ["Project", "AfriHealth AI"],
        ["Model", str(metadata.get("selected_model", "Not specified"))],
        ["Dataset", str(metadata.get("dataset", "Not specified"))],
        ["Evaluation records", str(record_count)],
        ["Evaluation basis", evaluation_basis],
        ["Project version", str(metadata.get("version", "Not specified"))],
        ["Intended use", str(metadata.get(
            "intended_use",
            "Research and educational prototype only.",
        ))],
    ]
    project_table = Table(project_rows, colWidths=[48 * mm, 112 * mm])
    project_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e7f4f4")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        project_table, Spacer(1, 14),
        Paragraph("Evaluation Metrics", styles["Heading2"]),
    ])

    metric_rows = [["Metric", "Value"]]
    metric_rows.extend([[name, f"{value:.4f}"] for name, value in metrics.items()])
    metric_table = Table(metric_rows, colWidths=[80 * mm, 50 * mm], repeatRows=1)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#087f8c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        metric_table, Spacer(1, 14),
        Paragraph("Interpretation and Limitations", styles["Heading2"]),
        Paragraph(
            "These metrics describe the evaluated sample only. They do not "
            "establish performance on independent data or in Nigerian or wider "
            "African populations. Predicted probabilities have not been "
            "established as clinically calibrated individual-risk estimates. "
            "A model output is not a diagnosis or treatment recommendation.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph(
            "Report dataset provenance, inclusion criteria, target definition, "
            "feature coding, missing-data handling, model-selection procedure, "
            "and uncertainty when presenting these findings. Do not claim "
            "external validation unless it has genuinely been performed.",
            styles["BodyText"],
        ),
    ])

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(18 * mm, 10 * mm, "AfriHealth AI | Research only")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# LOAD MODEL AND METADATA
# ============================================================

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
        "the feature order expected by app.py."
    )
    st.stop()

selected_model = metadata.get("selected_model", "Not recorded")
cv_auc = metadata.get(
    "cross_validation_roc_auc",
    metadata.get("cross_validation_auc"),
)
try:
    cv_auc_display = f"{float(cv_auc):.4f}" if cv_auc is not None else "Not recorded"
except (ValueError, TypeError):
    cv_auc_display = str(cv_auc)

training_records = metadata.get("training_records", "N/A")
test_records = metadata.get("test_records", "N/A")


# ============================================================
# HEADER AND PROJECT OVERVIEW
# ============================================================

st.title("🫀 AfriHealth AI")
st.subheader("AI-powered health-risk intelligence for Africa")
st.write(
    "A research and educational prototype for exploring "
    "machine-learning-based cardiovascular classification."
)
st.info(
    "**Research and educational use only.** This prototype is not a "
    "medical diagnostic tool. Its model uses the UCI Heart Disease dataset "
    "and has not been externally validated on Nigerian or broader African "
    "populations."
)

st.divider()
st.header("📊 Project Overview")
overview_cols = st.columns(4)
overview_cols[0].metric("Selected model", str(selected_model))
overview_cols[1].metric("CV ROC-AUC", cv_auc_display)
overview_cols[2].metric("Training records", str(training_records))
overview_cols[3].metric("Holdout records", str(test_records))
st.caption(
    "Cross-validation ROC-AUC is a model-selection metric. It is not an "
    "individual's disease probability or a guarantee of clinical performance."
)


# ============================================================
# RESEARCH INPUT FORM AND PREDICTION
# ============================================================

st.divider()
st.header("🩺 Research Input")
st.write("Enter values using the coding scheme expected by the saved model.")

with st.form("risk_prediction_form"):
    st.subheader("Demographic Information")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    with c2:
        sex = st.selectbox(
            "Sex coding",
            options=[0, 1],
            format_func=lambda value: "Female (0)" if value == 0 else "Male (1)",
        )

    st.subheader("Clinical Measurements")
    c1, c2 = st.columns(2)
    with c1:
        trestbps = st.number_input(
            "Resting Blood Pressure (trestbps)",
            min_value=50, max_value=250, value=120, step=1,
        )
        chol = st.number_input(
            "Serum Cholesterol (chol)",
            min_value=50, max_value=700, value=200, step=1,
        )
        thalach = st.number_input(
            "Maximum Heart Rate (thalach)",
            min_value=50, max_value=250, value=150, step=1,
        )
    with c2:
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
        )

    st.subheader("Diagnostic Features")
    c1, c2 = st.columns(2)
    with c1:
        cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        exang = st.selectbox(
            "Exercise-Induced Angina (exang)",
            options=[0, 1],
            format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
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
if submitted:
    input_data = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal,
    }], columns=DEFAULT_FEATURES)

    try:
        prediction = int(model.predict(input_data)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            class_labels = list(model.classes_)
            if 1 in class_labels:
                probability = float(
                    model.predict_proba(input_data)[0][class_labels.index(1)]
                )
    except Exception as error:
        st.error("The model could not process these inputs.")
        st.exception(error)
        prediction = None
        probability = None

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
            st.write("Probability output is not available.")

        st.warning(
            "This is a model classification, not a diagnosis. The model's "
            "probability is not a clinically validated estimate of personal "
            "disease risk. Do not use this output for medical or treatment decisions."
        )
        with st.expander("View submitted research inputs"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)


# ============================================================
# BUILD 04 - INTERNAL HOLDOUT EVALUATION DASHBOARD
# ============================================================

st.divider()
st.header("📈 Model Evaluation Dashboard")
st.caption(
    "Metrics in this section use exported holdout predictions and describe "
    "performance on that dataset only."
)

evaluation_df = None
evaluation_actual = evaluation_predicted = evaluation_probability = None

if not EVALUATION_PATH.exists():
    st.info(
        "Add evaluation_predictions.csv beside app.py to enable the "
        "evaluation dashboard and report downloads."
    )
else:
    try:
        evaluation_df = pd.read_csv(EVALUATION_PATH)
        evaluation_actual, evaluation_predicted, evaluation_probability = (
            validate_binary_evaluation_frame(
                evaluation_df, {"y_true", "y_pred", "prob_class_1"}
            )
        )

        basic_metrics = {
            "Accuracy": accuracy_score(evaluation_actual, evaluation_predicted),
            "Precision": precision_score(
                evaluation_actual, evaluation_predicted, zero_division=0
            ),
            "Recall / sensitivity": recall_score(
                evaluation_actual, evaluation_predicted, zero_division=0
            ),
            "F1 score": f1_score(
                evaluation_actual, evaluation_predicted, zero_division=0
            ),
        }
        metric_cols = st.columns(4)
        for col, (name, value) in zip(metric_cols, basic_metrics.items()):
            col.metric(name, f"{value:.3f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(
            evaluation_actual, evaluation_predicted, labels=[0, 1]
        )
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[[0, "#102235"], [0.5, "#087f8c"], [1, "#5fffe9"]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
            showscale=False,
        ))
        cm_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
            height=380,
            margin=dict(l=30, r=20, t=20, b=40),
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        if len(np.unique(evaluation_actual)) == 2:
            roc_fig, auc_value = make_roc_figure(
                evaluation_actual, evaluation_probability, "Holdout ROC Curve"
            )
            st.plotly_chart(roc_fig, use_container_width=True)
            st.metric("Holdout ROC-AUC", f"{auc_value:.3f}")

        summary = pd.DataFrame({
            "Metric": list(basic_metrics.keys()),
            "Value": list(basic_metrics.values()),
        })
        if len(np.unique(evaluation_actual)) == 2:
            summary.loc[len(summary)] = [
                "Holdout ROC-AUC",
                roc_auc_score(evaluation_actual, evaluation_probability),
            ]
        st.subheader("Evaluation Summary")
        st.dataframe(
            summary.style.format({"Value": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Evaluation rows: {len(evaluation_df)}. These results do not "
            "establish clinical validity or generalisation to Nigeria."
        )
    except Exception as error:
        st.error("The evaluation dashboard could not load the CSV.")
        st.caption(str(error))
        evaluation_df = None
        evaluation_actual = evaluation_predicted = evaluation_probability = None


# ============================================================
# BUILD 05A - RELIABILITY, CALIBRATION, AND CLASSIFICATION DETAILS
# ============================================================

st.divider()
st.header("🛡️ Model Reliability and Validation")

if evaluation_df is None:
    st.info("Reliability checks require a valid evaluation_predictions.csv.")
else:
    try:
        if len(np.unique(evaluation_actual)) < 2:
            st.warning(
                "Both actual classes are required for the full reliability analysis."
            )
        else:
            reliability_metrics = binary_metrics(
                evaluation_actual, evaluation_predicted, evaluation_probability
            )
            reliability_cols = st.columns(4)
            reliability_display = [
                ("Balanced accuracy", reliability_metrics["Balanced accuracy"]),
                ("Sensitivity", reliability_metrics["Sensitivity"]),
                ("Specificity", reliability_metrics["Specificity"]),
                ("Brier score", reliability_metrics["Brier score"]),
            ]
            for col, (name, value) in zip(reliability_cols, reliability_display):
                col.metric(name, f"{value:.3f}")

            st.caption(
                "Sensitivity is the fraction of class-1 cases identified. "
                "Specificity is the fraction of class-0 cases identified. "
                "Lower Brier scores indicate better probability accuracy on "
                "this sample; they do not prove clinical calibration."
            )

            st.subheader("Probability Calibration Check")
            calibration_fig = make_calibration_figure(
                evaluation_actual, evaluation_probability, bins=5, strategy="uniform"
            )
            st.plotly_chart(calibration_fig, use_container_width=True)
            st.caption(
                "Calibration bins can be unstable with a small holdout sample. "
                "This diagnostic does not prove that probabilities are calibrated."
            )

            st.subheader("Classification Details")
            report = classification_report(
                evaluation_actual,
                evaluation_predicted,
                labels=[0, 1],
                target_names=["Class 0", "Class 1"],
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).transpose().style.format("{:.3f}"),
                use_container_width=True,
            )

            with st.expander("Reliability limitations"):
                st.markdown("""
                - This is a limited holdout sample from the UCI Heart Disease dataset.
                - Results can change with a different sample or data split.
                - Calibration may not transfer to other hospitals or populations.
                - These checks do not constitute independent Nigerian or African validation.
                - Do not use this prototype for diagnosis or treatment decisions.
                """)

    except Exception as error:
        st.error("Reliability checks could not be completed.")
        st.caption(str(error))


# ============================================================
# BUILD 05B - PDF AND CSV RESEARCH REPORTS
# ============================================================

st.divider()
st.header("📄 Downloadable Research Reports")

if evaluation_df is None:
    st.info("A valid evaluation_predictions.csv is required for report downloads.")
else:
    try:
        report_metrics = binary_metrics(
            evaluation_actual, evaluation_predicted, evaluation_probability
        )
        report_metadata = metadata
        pdf_bytes = create_afrihealth_pdf(
            report_metrics,
            report_metadata,
            len(evaluation_actual),
            "Internal holdout predictions",
        )
        st.download_button(
            "Download PDF Research Report",
            data=pdf_bytes,
            file_name="AfriHealth_AI_Research_Report.pdf",
            mime="application/pdf",
        )
        st.download_button(
            "Download Evaluation Predictions CSV",
            data=evaluation_df.to_csv(index=False).encode("utf-8"),
            file_name="AfriHealth_AI_Evaluation_Predictions.csv",
            mime="text/csv",
        )
        st.caption(
            "Inspect the CSV before sharing it. It includes true labels and "
            "predictions from the holdout evaluation."
        )
    except Exception as error:
        st.error("The research report could not be generated.")
        st.caption(str(error))


# ============================================================
# BUILD 04B - MODEL EXPLAINABILITY
# ============================================================

st.divider()
st.header("🔎 Model Explainability")
st.write(
    "Explore transformed features associated with the model's output. "
    "These explanations describe model behaviour, not medical causes."
)
st.caption(
    "Coefficient magnitude is not the same as clinical importance. "
    "Preprocessing, scaling, encoding, and correlated features affect interpretation."
)

try:
    if not hasattr(model, "steps") or len(model.steps) < 2:
        raise ValueError("The saved model is not a supported preprocessing pipeline.")

    estimator = model.steps[-1][1]
    preprocessing = model[:-1]
    if not hasattr(estimator, "coef_"):
        raise ValueError("Coefficient-based explanations require a compatible estimator.")
    if not hasattr(preprocessing, "get_feature_names_out"):
        raise ValueError("The preprocessing pipeline cannot return transformed feature names.")

    transformed_names = preprocessing.get_feature_names_out()
    estimator_classes = list(estimator.classes_)
    if 1 not in estimator_classes:
        raise ValueError("The model does not contain positive class 1.")
    coefficients = np.asarray(estimator.coef_)
    if coefficients.shape[0] != 1 or len(estimator_classes) != 2:
        raise ValueError("This section supports binary Logistic Regression only.")

    positive_class_index = estimator_classes.index(1)
    coef_class_1 = coefficients[0] if positive_class_index == 1 else -coefficients[0]
    if len(transformed_names) != len(coef_class_1):
        raise ValueError("Feature names and coefficient counts do not match.")

    importance_df = pd.DataFrame({
        "Feature": [str(name).replace("__", " / ") for name in transformed_names],
        "Coefficient": coef_class_1,
        "Absolute coefficient": np.abs(coef_class_1),
    }).sort_values("Absolute coefficient", ascending=False)

    with st.expander("How to read these explanations"):
        st.markdown("""
        - **Global view:** ranks transformed features by absolute coefficient magnitude.
        - **Positive coefficient:** an increase in that transformed feature pushes class-1 log-odds upward.
        - **Negative coefficient:** an increase pushes class-1 log-odds downward.
        - **Local view:** shows signed feature contributions for the submitted record.
        """)
        st.caption("Categorical variables may be expanded into several encoded columns.")

    st.subheader("1. Global Feature Influence")
    top_global = importance_df.head(15).sort_values("Absolute coefficient", ascending=True)
    global_fig = go.Figure(go.Bar(
        x=top_global["Absolute coefficient"],
        y=top_global["Feature"],
        orientation="h",
        marker_color="#16c7b7",
        hovertemplate="Feature: %{y}<br>Absolute coefficient: %{x:.4f}<extra></extra>",
    ))
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
        importance_df.head(15)[["Feature", "Coefficient", "Absolute coefficient"]]
        .style.format({"Coefficient": "{:.4f}", "Absolute coefficient": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("2. Individual Prediction Explanation")
    if submitted and input_data is not None:
        transformed_input = preprocessing.transform(input_data)
        if hasattr(transformed_input, "toarray"):
            transformed_input = transformed_input.toarray()
        transformed_row = np.asarray(transformed_input)[0]
        contributions = transformed_row * coef_class_1

        intercept = float(np.asarray(estimator.intercept_).ravel()[0])
        if positive_class_index == 0:
            intercept = -intercept

        local_df = pd.DataFrame({
            "Feature": [str(name).replace("__", " / ") for name in transformed_names],
            "Contribution": contributions,
        })
        local_df["Absolute contribution"] = local_df["Contribution"].abs()
        local_df = local_df.sort_values("Absolute contribution", ascending=False)

        st.write(
            "The chart shows the largest transformed-feature contributions "
            "to class-1 log-odds for the submitted record."
        )
        top_local = local_df.head(12).sort_values("Contribution")
        directions = np.where(
            top_local["Contribution"] >= 0,
            "Pushes towards class 1",
            "Pushes away from class 1",
        )
        local_fig = go.Figure(go.Bar(
            x=top_local["Contribution"],
            y=top_local["Feature"],
            orientation="h",
            marker_color=[
                "#ff8c69" if value >= 0 else "#16c7b7"
                for value in top_local["Contribution"]
            ],
            customdata=np.array(directions).reshape(-1, 1),
            hovertemplate=(
                "Feature: %{y}<br>Contribution: %{x:.4f}<br>"
                "%{customdata[0]}<extra></extra>"
            ),
        ))
        local_fig.add_vline(x=0, line_dash="dash", line_color="#9fb0c0")
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
            "Positive contributions increase class-1 log-odds; negative "
            "contributions decrease them. These are not percentage points "
            "or individual medical risk factors."
        )
        with st.expander("View contribution details"):
            st.dataframe(
                local_df[["Feature", "Contribution", "Absolute contribution"]]
                .style.format({"Contribution": "{:.4f}", "Absolute contribution": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        st.metric("Model intercept for class-1 log-odds", f"{intercept:.4f}")
    else:
        st.info("Submit the Research Input form to view an individual explanation.")

except Exception as error:
    st.warning(
        "The explainability section could not be generated. Other sections "
        "may still work if the model is compatible."
    )
    st.caption(f"Technical detail: {error}")


# ============================================================
# MODEL INFORMATION, FEATURE SCHEMA, AND LIMITATIONS
# ============================================================

st.divider()
st.header("🔬 Model Information")
with st.expander("View model and research details"):
    st.write("Selected model:", selected_model)
    st.write("Selection metric:", metadata.get("selection_metric", "Not recorded"))
    st.write("Cross-validation ROC-AUC:", cv_auc_display)
    st.write("Dataset:", metadata.get("dataset", "Not recorded"))
    st.write("Training records:", training_records)
    st.write("Holdout test records:", test_records)
    st.write("Scikit-learn version:", metadata.get("scikit_learn_version", "Not recorded"))
    st.write("Model version:", metadata.get("version", "Not recorded"))
    st.write("Created (UTC):", metadata.get("created_at_utc", "Not recorded"))
    st.write("Intended use:", metadata.get(
        "intended_use", "Research and educational use only."
    ))

with st.expander("📚 View input feature schema"):
    feature_table = pd.DataFrame({
        "Feature": DEFAULT_FEATURES,
        "Purpose": [
            "Age", "Sex coding", "Chest pain type", "Resting blood pressure",
            "Serum cholesterol", "Fasting blood sugar", "Resting ECG",
            "Maximum heart rate", "Exercise-induced angina", "ST depression",
            "ST segment slope", "Major vessels", "Thalassemia",
        ],
    })
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

with st.expander("⚠️ Research limitations"):
    st.markdown("""
    **Dataset:** UCI Heart Disease dataset, repository ID 45.

    **Sample size:** 303 records in the source dataset, with 242 used for
    training and 61 reserved for holdout testing according to the project metadata.

    **Model selection:** Logistic Regression was selected using mean stratified
    cross-validation ROC-AUC.

    **Population limitation:** The dataset is not representative of Nigeria or
    the wider African population.

    **External validation:** No independent Nigerian or African clinical
    validation is established by this app alone.

    **Clinical use:** Do not use this prototype for diagnosis, treatment,
    patient triage, or clinical decision-making.

    **Future research:** Representative data, independent validation,
    calibration, fairness evaluation, privacy safeguards, and clinical
    collaboration are needed.
    """)


# ============================================================
# BUILD 06A / 06B - EXTERNAL VALIDATION WORKSPACE
# ============================================================

st.divider()
st.header("🧪 Build 06: External Validation Workspace")
st.write(
    "Evaluate the existing saved model on a separately sourced dataset. "
    "This workflow does not retrain the model."
)
st.caption(
    "The CSV must contain the 13 feature columns below plus a binary "
    "`target` column coded 0 or 1. Confirm that definitions, units, category "
    "codes, target definition, permissions, and independence match the study plan."
)
st.code(", ".join(EXTERNAL_FEATURES + ["target"]))

external_file = st.file_uploader(
    "Upload an authorised, de-identified validation CSV",
    type=["csv"],
    key="external_validation_csv",
)
external_source = st.text_input(
    "Dataset source or study identifier",
    placeholder="Example: independent cohort, study identifier",
    key="external_validation_source",
)
st.warning(
    "Do not upload names, addresses, phone numbers, medical record numbers, "
    "or other directly identifying information. Ensure you have permission "
    "to use the dataset."
)

if external_file is not None and st.button(
    "Run validation on uploaded dataset",
    key="run_external_validation",
    type="primary",
):
    try:
        external_df = pd.read_csv(external_file)
        required = set(EXTERNAL_FEATURES + ["target"])
        missing = required - set(external_df.columns)

        if missing:
            st.error("Missing required columns: " + ", ".join(sorted(missing)))
        elif external_df.empty:
            st.error("The uploaded dataset contains no records.")
        else:
            X_external = external_df[EXTERNAL_FEATURES].apply(
                pd.to_numeric, errors="coerce"
            )
            y_series = pd.to_numeric(external_df["target"], errors="coerce")

            bad_features = X_external.columns[X_external.isnull().any()].tolist()
            if bad_features:
                st.error(
                    "Missing or non-numeric values in feature columns: "
                    + ", ".join(bad_features)
                )
            elif y_series.isnull().any():
                st.error("The target contains missing or non-numeric values.")
            elif not np.isfinite(X_external.to_numpy(dtype=float)).all():
                st.error("Feature values must all be finite numbers.")
            elif not np.isin(y_series.to_numpy(), [0, 1]).all():
                st.error("Target values must be binary 0 or 1.")
            elif y_series.nunique() < 2:
                st.error("Both target classes must be present.")
            elif not hasattr(model, "predict_proba"):
                st.error("The saved model does not support predict_proba.")
            elif not hasattr(model, "classes_") or 1 not in list(model.classes_):
                st.error("The saved model does not expose the expected positive class 1.")
            else:
                y_external = y_series.to_numpy(dtype=int)

                # Use only the frozen model; do not fit or tune it here.
                external_predictions = np.asarray(
                    model.predict(X_external), dtype=int
                )
                class_index = list(model.classes_).index(1)
                external_probabilities = np.asarray(
                    model.predict_proba(X_external)[:, class_index],
                    dtype=float,
                )

                if not np.isin(external_predictions, [0, 1]).all():
                    st.error("The model returned labels other than 0 and 1.")
                elif (
                    not np.isfinite(external_probabilities).all()
                    or ((external_probabilities < 0) |
                        (external_probabilities > 1)).any()
                ):
                    st.error("The model produced invalid probability values.")
                else:
                    ext_metrics = binary_metrics(
                        y_external, external_predictions, external_probabilities
                    )

                    # Class-stratified bootstrap confidence intervals.
                    rng = np.random.default_rng(2026)
                    class_zero = np.flatnonzero(y_external == 0)
                    class_one = np.flatnonzero(y_external == 1)
                    boot = {name: [] for name in ext_metrics}

                    for _ in range(1000):
                        sample_idx = np.concatenate([
                            rng.choice(class_zero, size=len(class_zero), replace=True),
                            rng.choice(class_one, size=len(class_one), replace=True),
                        ])
                        boot_metrics = binary_metrics(
                            y_external[sample_idx],
                            external_predictions[sample_idx],
                            external_probabilities[sample_idx],
                        )
                        for name in boot:
                            boot[name].append(boot_metrics[name])

                    ci_rows = []
                    for name, estimate in ext_metrics.items():
                        lower, upper = np.percentile(boot[name], [2.5, 97.5])
                        ci_rows.append({
                            "Metric": name,
                            "Estimate": estimate,
                            "95% CI lower": lower,
                            "95% CI upper": upper,
                        })
                    ci_df = pd.DataFrame(ci_rows)

                    st.subheader("Validation Results")
                    st.write(
                        "Dataset source: "
                        + (external_source.strip() or "Not supplied")
                    )
                    st.write(f"Records evaluated: {len(y_external)}")
                    st.write(
                        f"Class 0: {len(class_zero)} | Class 1: {len(class_one)}"
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
                        "Intervals are percentile intervals from 1,000 class-stratified "
                        "bootstrap replicates. They reflect sampling variation under "
                        "this procedure and not all sources of uncertainty."
                    )

                    st.subheader("External Dataset ROC Curve")
                    ext_roc_fig, _ = make_roc_figure(
                        y_external, external_probabilities
                    )
                    st.plotly_chart(ext_roc_fig, use_container_width=True)

                    st.subheader("External Dataset Probability Calibration")
                    ext_calibration_fig = make_calibration_figure(
                        y_external, external_probabilities,
                        bins=5, strategy="quantile",
                    )
                    st.plotly_chart(ext_calibration_fig, use_container_width=True)
                    st.caption(
                        "Calibration estimates may be unstable with small samples. "
                        "They do not establish clinically valid individual-risk estimates."
                    )

                    results_df = X_external.copy()
                    results_df["y_true"] = y_external
                    results_df["y_pred"] = external_predictions
                    results_df["prob_class_1"] = external_probabilities
                    results_df["dataset_source"] = external_source.strip() or "Not supplied"

                    st.subheader("Download Validation Results")
                    st.download_button(
                        "Download predictions CSV",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="AfriHealth_Validation_Predictions.csv",
                        mime="text/csv",
                        key="download_external_predictions",
                    )
                    st.download_button(
                        "Download metrics and confidence intervals",
                        data=ci_df.to_csv(index=False).encode("utf-8"),
                        file_name="AfriHealth_Validation_Metrics.csv",
                        mime="text/csv",
                        key="download_external_metrics",
                    )

                    st.warning(
                        "Call this external validation only after confirming dataset "
                        "independence, provenance, permissions, and compatible feature "
                        "definitions. Uploading data does not establish clinical validity."
                    )

    except Exception as error:
        st.error(
            "Validation could not be completed. Check the CSV schema, feature "
            "coding, target labels, data permissions, and model compatibility."
        )
        st.caption(str(error))


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "AfriHealth AI | AI-powered health-risk intelligence for Africa | "
    "Research and educational prototype | Version 0.1.0"
)
'''

out = Path("/mnt/data/app.py")
out.write_text(app_code, encoding="utf-8")
print(f"Created {out} ({len(app_code.splitlines())} lines, {out.stat().st_size:,} bytes)")
