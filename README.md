# Sales Model - Random Forest

## Model
Random Forest Classifier dengan TF-IDF untuk memprediksi `Category Name` dari `Item Name`.

Input yang digunakan sengaja hanya `Item Name`. `Category Code` dan `Category Name` tidak digunakan sebagai fitur karena merupakan target/representasi target dan dapat menyebabkan data leakage.

## Evaluasi
- Accuracy: 0.9020
- Weighted Precision: 0.9220
- Weighted Recall: 0.9020
- Weighted F1: 0.9026

## Menjalankan frontend
```bash
pip install -r requirements.txt
streamlit run app.py
```

## File model
- `sales_model.pkl`
- `sales_model.pkt` (salinan dengan ekstensi yang diminta)
