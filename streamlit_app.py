import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model dan scaler yang sama saat training
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Pastikan scaler yang sama

# Fitur yang digunakan dalam model
all_features = [
    'SafetySecurity', 'PersonelFreedom', 'Governance',
    'SocialCapital', 'InvestmentEnvironment', 'EnterpriseConditions',
    'MarketAccessInfrastructure', 'EconomicQuality', 'LivingConditions',
    'Health', 'Education', 'NaturalEnvironment'
]

# Tampilan GUI
st.set_page_config(page_title="Prediksi Cluster Negara", layout="centered")
st.title("🌍 Prediksi Cluster Negara")
st.write("""
Masukkan nilai indikator ekonomi negara berdasarkan skala 0-10, untuk mengetahui cluster ekonomi negara.
""")

# Input data pengguna
st.sidebar.header("📊 Input Nilai Indikator")
input_data = {feature: 0.0 for feature in all_features}

for feature in all_features:
    input_data[feature] = st.sidebar.slider(
        f"{feature}", min_value=0.0, max_value=10.0, step=0.1, value=5.0
    )

# Convert input menjadi DataFrame
input_df = pd.DataFrame([input_data])

# Standarisasi input dengan scaler yang sama
scaled_input = scaler.transform(input_df)

# Prediksi cluster
predicted_cluster = kmeans_model.predict(scaled_input)[0]

# Tampilkan hasil prediksi
st.subheader("📋 Hasil Prediksi")
st.markdown(f"""
### Negara yang dimasukkan termasuk dalam **Cluster {predicted_cluster}**.
#### Penjelasan Cluster:
- **Cluster 0**: Negara ekonomi tertinggal
- **Cluster 1**: Negara ekonomi berkembang
- **Cluster 2**: Negara ekonomi maju
""")

# Tambahkan tabel input pengguna untuk visualisasi
st.write("#### Nilai Indikator yang Dimasukkan")
st.table(input_df)

# Footer
st.markdown("---")
st.markdown("Created by **[Nama Anda]** - Machine Learning Application")
