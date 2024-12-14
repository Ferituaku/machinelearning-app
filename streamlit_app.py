import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load Model dan Scaler
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Scaler yang sama saat training

# Definisikan Semua Fitur
all_features = [
    'SafetySecurity', 'PersonelFreedom', 'Governance',
    'SocialCapital', 'InvestmentEnvironment', 'EnterpriseConditions',
    'MarketAccessInfrastructure', 'EconomicQuality', 'LivingConditions',
    'Health', 'Education', 'NaturalEnvironment'
]
important_features = ['SafetySecurity', 'Governance', 'EconomicQuality', 'LivingConditions']

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Cluster Negara", layout="centered")

# Judul Aplikasi
st.title("🌍 Prediksi Cluster Negara")
st.write("""
Masukkan nilai indikator ekonomi negara berdasarkan skala 0-10, untuk mengetahui cluster ekonomi negara.
""")

# Sidebar untuk Input Data
st.sidebar.header("📊 Input Nilai Indikator")
input_data = {feature: 0.0 for feature in all_features}  # Inisialisasi semua fitur dengan 0.0
for feature in important_features:
    input_data[feature] = st.sidebar.slider(
        f"{feature}",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=5.0,
        help=f"Masukkan nilai indikator untuk {feature} (0-10)"
    )

# Konversi Input ke DataFrame
input_df = pd.DataFrame([input_data])

# Pastikan Input Memiliki Fitur yang Sama
input_df = input_df[all_features]  # Urutkan kolom sesuai dengan data pelatihan

# Standarisasi Input
scaled_input = scaler.transform(input_df)

# Prediksi Cluster
predicted_cluster = kmeans_model.predict(scaled_input)[0]

# Hasil Prediksi
st.subheader("📋 Hasil Prediksi")
st.success(f"Negara yang dimasukkan termasuk dalam **Cluster {predicted_cluster}**.")
st.write("""
Penjelasan Cluster:
- **Cluster 0**: Negara ekonomi tertinggal
- **Cluster 1**: Negara ekonomi berkembang
- **Cluster 2**: Negara ekonomi maju
""")

# Tampilkan Tabel Input
st.write("#### 📋 Nilai Indikator yang Dimasukkan")
st.table(input_df)

# Visualisasi Input
st.write("#### 📊 Visualisasi Indikator yang Dimasukkan")
st.bar_chart(input_df.T)

# Footer
st.markdown("---")
st.markdown("✨ Created by **[Nama Anda]** - Machine Learning Application")
