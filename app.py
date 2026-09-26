
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sales Random Forest | UAS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS
# =========================================================
ROOT = Path(__file__).resolve().parent

MODEL_PATHS = [
    ROOT / "sales_model.pkl",
    ROOT / "sales_model.pkt",
]

DATA_PATHS = [
    ROOT / "data" / "annex1_cleaned_preprocessed.csv",
    ROOT / "annex1_cleaned_preprocessed.csv",
]

METRICS_PATH = ROOT / "metrics.json"
REPORT_PATH = ROOT / "classification_report.txt"
CM_PATH = ROOT / "confusion_matrix.csv"

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.stApp {
    background: #f5f7fb;
}
.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #111827;
}
[data-testid="stSidebar"] * {
    color: #f3f4f6 !important;
}
.hero {
    background: linear-gradient(135deg, #111827, #1f2937);
    border-radius: 24px;
    padding: 2rem 2.2rem;
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 14px 35px rgba(15,23,42,.12);
}
.hero h1 {
    margin: 0;
    font-size: 2.35rem;
    font-weight: 850;
    letter-spacing: -0.04em;
}
.hero p {
    margin: .55rem 0 0;
    color: #d1d5db;
    font-size: 1rem;
}
.badge {
    display: inline-block;
    padding: .28rem .65rem;
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    color: #e5e7eb;
    font-size: .78rem;
    font-weight: 700;
    margin-bottom: .7rem;
}
.kpi {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1rem 1.15rem;
    box-shadow: 0 6px 20px rgba(15,23,42,.05);
}
.kpi-label {
    color: #6b7280;
    font-size: .76rem;
    font-weight: 750;
    text-transform: uppercase;
    letter-spacing: .05em;
}
.kpi-value {
    color: #111827;
    font-size: 1.55rem;
    font-weight: 850;
    margin-top: .25rem;
}
.result {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 1.35rem;
    box-shadow: 0 8px 26px rgba(15,23,42,.06);
}
.result-label {
    color: #6b7280;
    font-size: .75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.result-value {
    color: #111827;
    font-size: 1.8rem;
    font-weight: 850;
    margin: .35rem 0;
}
.small-muted {
    color: #6b7280;
    font-size: .88rem;
}
.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 2.5rem;
    font-size: .8rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_resource
def load_model():
    path = first_existing(MODEL_PATHS)
    if path is None:
        return None, None
    try:
        return joblib.load(path), path
    except Exception as exc:
        return f"ERROR::{exc}", path


@st.cache_data
def load_dataset():
    path = first_existing(DATA_PATHS)
    if path is None:
        return None, None
    try:
        df = pd.read_csv(path)
        return df, path
    except Exception:
        return None, path


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_classification_report():
    if not REPORT_PATH.exists():
        return None
    try:
        return REPORT_PATH.read_text(encoding="utf-8")
    except Exception:
        return None


@st.cache_data
def load_confusion_matrix():
    if not CM_PATH.exists():
        return None
    try:
        return pd.read_csv(CM_PATH, index_col=0)
    except Exception:
        try:
            return pd.read_csv(CM_PATH)
        except Exception:
            return None


def metric_value(metrics, key, fallback):
    value = metrics.get(key, fallback)
    try:
        return float(value)
    except Exception:
        return fallback


def predict(model, product_name):
    pred = model.predict([product_name])[0]

    probs = None
    classes = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([product_name])[0]
        classes = getattr(model, "classes_", None)

    # For a Pipeline, classes_ normally exists on the final estimator.
    if classes is None and hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        if steps:
            classes = getattr(steps[-1], "classes_", None)

    return str(pred), probs, classes


def parse_report(text):
    """Parse the sklearn classification_report text into a dataframe when possible."""
    if not text:
        return None

    rows = []
    pattern = re.compile(
        r"^\s*(.+?)\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"([0-9]*\.?[0-9]+)\s+"
        r"(\d+)\s*$"
    )

    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        label = m.group(1).strip()
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append({
            "Category": label,
            "Precision": float(m.group(2)),
            "Recall": float(m.group(3)),
            "F1-Score": float(m.group(4)),
            "Support": int(m.group(5)),
        })

    if not rows:
        return None

    return pd.DataFrame(rows)


# =========================================================
# LOAD RESOURCES
# =========================================================
model, model_path = load_model()
df, data_path = load_dataset()
metrics = load_metrics()
report_text = load_classification_report()
cm = load_confusion_matrix()
report_df = parse_report(report_text)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 📊 Sales AI")
    st.caption("UAS — Implementasi CRISP-DM")
    st.divider()

    if model is not None and not isinstance(model, str):
        st.success("Model siap digunakan")
    elif model is None:
        st.error("Model tidak ditemukan")
    else:
        st.error("Model gagal dimuat")

    st.markdown("### Pipeline")
    st.markdown("**Input**  \nNama Produk")
    st.markdown("↓")
    st.markdown("**TF-IDF**  \nText Vectorization")
    st.markdown("↓")
    st.markdown("**Random Forest**  \n400 Trees")
    st.markdown("↓")
    st.markdown("**Output**  \nKategori Produk")

    st.divider()
    st.markdown("### CRISP-DM")
    st.markdown("1. Business Understanding")
    st.markdown("2. Data Understanding")
    st.markdown("3. Data Preparation")
    st.markdown("4. Modeling")
    st.markdown("5. Evaluation")
    st.markdown("6. Deployment")

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="badge">MACHINE LEARNING • CRISP-DM • UAS</div>
    <h1>Sales Category Predictor</h1>
    <p>
        Dashboard klasifikasi kategori produk menggunakan
        TF-IDF dan Random Forest.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPI
# =========================================================
accuracy = metric_value(metrics, "accuracy", 0.9020)
precision = metric_value(metrics, "weighted_precision", 0.9220)
recall = metric_value(metrics, "weighted_recall", 0.9020)
f1 = metric_value(metrics, "weighted_f1", 0.9026)

cols = st.columns(4)
for col, label, value in zip(
    cols,
    ["Accuracy", "Precision", "Recall", "F1 Score"],
    [accuracy, precision, recall, f1],
):
    with col:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# =========================================================
# TABS
# =========================================================
tab_pred, tab_data, tab_eval, tab_crisp = st.tabs([
    "🔎 Prediksi",
    "📊 Data Understanding",
    "📈 Evaluation",
    "🔄 CRISP-DM"
])

# =========================================================
# PREDICTION TAB
# =========================================================
with tab_pred:
    st.subheader("Prediksi Kategori Produk")
    st.caption("Masukkan nama produk. Model yang sudah ditraining akan dimuat dari file `.pkl` tanpa training ulang.")

    left, right = st.columns([1.05, .95], gap="large")

    with left:
        product_name = st.text_input(
            "Nama produk",
            placeholder="Contoh: Chinese Cabbage",
        )

        if df is not None and "Item Name" in df.columns:
            sample_options = (
                df["Item Name"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(12)
                .tolist()
            )
        else:
            sample_options = [
                "Chinese Cabbage",
                "Broccoli",
                "Needle Mushroom (Bag)",
                "Eggplant",
                "Red Pepper (1)",
            ]

        selected = st.selectbox(
            "Atau pilih contoh produk",
            ["— Pilih contoh —"] + sample_options,
        )

        if selected != "— Pilih contoh —" and not product_name:
            product_name = selected

        predict_clicked = st.button(
            "🚀 Prediksi Sekarang",
            type="primary",
            use_container_width=True,
        )

    with right:
        if model is None:
            st.warning("File `sales_model.pkl` belum ditemukan di repository.")
        elif isinstance(model, str):
            st.error("Model ditemukan tetapi gagal dimuat.")
            st.code(model.replace("ERROR::", ""))
        else:
            st.info(
                f"Model aktif: **{model_path.name}**  \n"
                "Model dimuat dari file hasil training."
            )

    if predict_clicked:
        if not product_name.strip():
            st.warning("Masukkan nama produk terlebih dahulu.")
        elif model is None or isinstance(model, str):
            st.error("Prediksi tidak dapat dilakukan karena model belum tersedia.")
        else:
            try:
                pred, probs, classes = predict(model, product_name.strip())

                st.markdown("### Hasil Prediksi")

                if probs is not None and classes is not None:
                    order = np.argsort(probs)[::-1]
                    top_n = min(5, len(order))

                    labels = [str(classes[i]) for i in order[:top_n]]
                    values = [float(probs[i]) for i in order[:top_n]]
                    confidence = values[0]

                    r1, r2 = st.columns([.85, 1.15], gap="large")

                    with r1:
                        st.markdown(
                            f"""
                            <div class="result">
                                <div class="result-label">Predicted category</div>
                                <div class="result-value">{pred}</div>
                                <div class="small-muted">
                                    Confidence: <b>{confidence:.2%}</b>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with r2:
                        chart = pd.DataFrame({
                            "Category": labels,
                            "Probability": values,
                        }).set_index("Category")
                        st.bar_chart(chart, horizontal=True)

                    st.markdown("#### Top probability")
                    prob_df = pd.DataFrame({
                        "Rank": range(1, top_n + 1),
                        "Category": labels,
                        "Probability": [f"{v:.2%}" for v in values],
                    })
                    st.dataframe(
                        prob_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success(f"Predicted category: **{pred}**")
                    st.caption("Model tidak menyediakan probabilitas prediksi.")

            except Exception as exc:
                st.error("Prediksi gagal.")
                st.code(str(exc))

# =========================================================
# DATA UNDERSTANDING TAB
# =========================================================
with tab_data:
    st.subheader("Data Understanding")

    if df is None:
        st.warning(
            "Dataset tidak ditemukan. Upload `annex1_cleaned_preprocessed.csv` "
            "ke repository atau ke folder `data/`."
        )
    else:
        target_col = "Category Name" if "Category Name" in df.columns else None
        item_col = "Item Name" if "Item Name" in df.columns else None

        total_rows = len(df)
        total_cols = len(df.columns)
        total_categories = df[target_col].nunique() if target_col else 0
        missing_total = int(df.isna().sum().sum())

        dcols = st.columns(4)
        for col, label, value in zip(
            dcols,
            ["Total Data", "Jumlah Kolom", "Kategori", "Missing Value"],
            [total_rows, total_cols, total_categories, missing_total],
        ):
            with col:
                st.metric(label, value)

        st.write("")
        c1, c2 = st.columns([1.05, .95], gap="large")

        with c1:
            st.markdown("#### Distribusi Kategori")
            if target_col:
                counts = (
                    df[target_col]
                    .dropna()
                    .value_counts()
                    .sort_values(ascending=False)
                    .rename("Jumlah")
                )
                st.bar_chart(counts, horizontal=True)
            else:
                st.info("Kolom `Category Name` tidak tersedia.")

        with c2:
            st.markdown("#### Statistik Dataset")
            stat_df = pd.DataFrame({
                "Statistik": [
                    "Jumlah data",
                    "Jumlah fitur mentah",
                    "Jumlah kategori target",
                    "Missing value",
                    "Duplikat",
                ],
                "Nilai": [
                    total_rows,
                    total_cols,
                    total_categories,
                    missing_total,
                    int(df.duplicated().sum()),
                ],
            })
            st.dataframe(stat_df, use_container_width=True, hide_index=True)

        st.markdown("#### Contoh Data")
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)

        st.markdown("#### Data Quality")
        quality = pd.DataFrame({
            "Kolom": df.columns,
            "Tipe Data": [str(x) for x in df.dtypes],
            "Missing": [int(x) for x in df.isna().sum()],
            "Unique": [int(x) for x in df.nunique(dropna=True)],
        })
        st.dataframe(quality, use_container_width=True, hide_index=True)

# =========================================================
# EVALUATION TAB
# =========================================================
with tab_eval:
    st.subheader("Model Evaluation")
    st.caption("Evaluasi model dari proses training/testing yang didokumentasikan pada notebook CRISP-DM.")

    e1, e2, e3, e4 = st.columns(4)
    for col, label, value in zip(
        [e1, e2, e3, e4],
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        [accuracy, precision, recall, f1],
    ):
        with col:
            st.metric(label, f"{value:.2%}")

    st.write("")

    eval_left, eval_right = st.columns(2, gap="large")

    with eval_left:
        st.markdown("#### Classification Report")
        if report_df is not None:
            display_df = report_df.copy()
            for c in ["Precision", "Recall", "F1-Score"]:
                display_df[c] = display_df[c].map(lambda x: f"{x:.2f}")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
        elif report_text:
            st.code(report_text, language="text")
        else:
            st.info("File `classification_report.txt` belum tersedia.")

    with eval_right:
        st.markdown("#### Confusion Matrix")
        if cm is not None:
            st.dataframe(cm, use_container_width=True)

            # Native Streamlit chart for a quick visual view.
            try:
                st.bar_chart(cm.sum(axis=1).rename("Actual samples"))
            except Exception:
                pass
        else:
            st.info("File `confusion_matrix.csv` belum tersedia.")

    st.markdown("#### Interpretasi")
    st.info(
        "Confusion matrix menunjukkan jumlah prediksi benar dan salah untuk setiap kelas. "
        "Classification report memperlihatkan precision, recall, F1-score, dan support per kategori."
    )

# =========================================================
# CRISP-DM TAB
# =========================================================
with tab_crisp:
    st.subheader("Implementasi CRISP-DM")

    stages = [
        ("1", "Business Understanding", "Menentukan masalah, tujuan, manfaat, dan target prediksi kategori produk."),
        ("2", "Data Understanding", "Memahami struktur dataset, atribut, distribusi kategori, missing value, dan kualitas data."),
        ("3", "Data Preparation", "Cleaning, pemilihan fitur, penghapusan duplikat, train-test split, dan TF-IDF."),
        ("4", "Modeling", "Pipeline TF-IDF + Random Forest dengan 400 trees."),
        ("5", "Evaluation", "Accuracy, precision, recall, F1-score, classification report, dan confusion matrix."),
        ("6", "Deployment", "Model disimpan sebagai `sales_model.pkl` dan dimuat oleh dashboard tanpa training ulang."),
    ]

    for number, title, description in stages:
        st.markdown(
            f"""
            <div class="result" style="margin-bottom:.7rem;">
                <div class="result-label">Tahap {number}</div>
                <div style="font-size:1.15rem;font-weight:800;color:#111827;margin:.25rem 0;">
                    {title}
                </div>
                <div class="small-muted">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Pipeline Machine Learning")
    st.code(
        "Item Name\n"
        "    ↓\n"
        "TF-IDF Vectorizer\n"
        "    ↓\n"
        "Random Forest (400 trees)\n"
        "    ↓\n"
        "Category Name",
        language="text",
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">Sales Random Forest • UAS Machine Learning • CRISP-DM</div>',
    unsafe_allow_html=True,
)
