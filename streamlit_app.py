import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model dan scaler
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Pastikan Anda menyimpan scaler saat melatih model

# Definisikan fitur
important_features = ['SafetySecurity', 'Governance', 'EconomicQuality', 'LivingConditions']

# Tampilan GUI
st.set_page_config(page_title="Prediksi Cluster Negara", layout="centered")
st.title("Prediksi Cluster Negara")
st.markdown("""
Masukkan nilai-nilai indikator ekonomi negara untuk mengetahui cluster-nya. 
Cluster ini dapat digunakan untuk analisis karakteristik negara berdasarkan data indikator utama.
""")

# Input data
st.sidebar.header("Masukkan Data Indikator")
input_data = {feature: 0.0 for feature in important_features}  # Inisialisasi semua fitur dengan nilai 0.0
for feature in important_features:
    input_data[feature] = st.sidebar.slider(
        feature,
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=5.0,
        help=f"Masukkan nilai untuk {feature} (0-10)"
    )

# Konversi input ke DataFrame
input_df = pd.DataFrame([input_data])

# Pastikan kolom input sesuai dengan data pelatihan
input_df = input_df[important_features]  # Urutkan kolom sesuai urutan yang digunakan saat pelatihan

# Scale input dengan scaler yang telah dilatih
scaled_input = scaler.transform(input_df)

# Prediksi cluster
predicted_cluster = kmeans_model.predict(scaled_input)[0]

# Tampilkan hasil prediksi
st.subheader("Hasil Prediksi")
st.success(f"Negara yang dimasukkan termasuk dalam **Cluster {predicted_cluster}**.")
st.write("""
Penjelasan Cluster:
- **Cluster 0**: Negara ekonomi tertinggal
- **Cluster 1**: Negara ekonomi berkembang
- **Cluster 2**: Negara ekonomi maju
""")

# Visualisasi data input
st.subheader("Visualisasi Input Data")
st.bar_chart(input_df.T)

# Tombol untuk reset
if st.button("Reset Data"):
    st.caching.clear_cache()
