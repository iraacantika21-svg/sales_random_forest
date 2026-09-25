import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Sales Category Predictor",
    page_icon="📊",
    layout="centered"
)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "sales_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📊 Sales Category Predictor")
st.write("Prediksi kategori produk menggunakan Random Forest + TF-IDF.")

item_name = st.text_input(
    "Nama Item",
    placeholder="Contoh: Fresh Carrot"
)

if st.button("Prediksi", type="primary"):
    if not item_name.strip():
        st.warning("Masukkan nama item terlebih dahulu.")
    else:
        prediction = model.predict([item_name.strip()])[0]

        st.success(f"Kategori prediksi: **{prediction}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([item_name.strip()])[0]
            classes = model.classes_
            result = pd.DataFrame({
                "Kategori": classes,
                "Probabilitas": proba
            }).sort_values("Probabilitas", ascending=False)

            st.subheader("Probabilitas")
            st.dataframe(
                result.style.format({"Probabilitas": "{:.2%}"}),
                use_container_width=True,
                hide_index=True
            )
