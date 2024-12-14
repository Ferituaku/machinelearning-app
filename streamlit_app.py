import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load Model dan Scaler
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Fitur Penting
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
input_data = {feature: 0.0 for feature in important_features}

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

# Debugging: Tampilkan Input Pengguna
st.write("#### 📋 Input Data Pengguna")
st.table(input_df)

# Standarisasi Input
scaled_input = scaler.transform(input_df)

# Debugging: Tampilkan Data yang Di-scale
st.write("#### 📊 Data Setelah Scaler")
st.table(pd.DataFrame(scaled_input, columns=important_features))

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

# Footer
st.markdown("---")
st.markdown("✨ Created by **[Nama Anda]** - Machine Learning Application")
