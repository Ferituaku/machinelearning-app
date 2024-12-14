import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Cluster Ekonomi Global",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        color: white;
    }
    .st-emotion-cache-1v0mbdj.e115fcil1 {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk validasi input
def validate_input(input_data):
    for feature, value in input_data.items():
        if value < 0 or value > 10:
            return False, f"Nilai {feature} harus antara 0 dan 10"
    return True, ""

# Inisialisasi session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Load model dan scaler
if not st.session_state.model_loaded:
    try:
        with st.spinner('Memuat model...'):
            model_path = 'kmeans_model.pkl'
            scaler_path = 'scaler.pkl'
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                st.error("File model tidak ditemukan! Pastikan file kmeans_model.pkl dan scaler.pkl tersedia.")
                st.stop()
            
            st.session_state.model = pickle.load(open(model_path, 'rb'))
            st.session_state.scaler = pickle.load(open(scaler_path, 'rb'))
            st.session_state.model_loaded = True
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        st.stop()

# Judul dan deskripsi aplikasi
st.title("🌍 Prediksi Cluster Ekonomi Negara")
st.markdown("""
Aplikasi ini memprediksi cluster ekonomi suatu negara berdasarkan indikator-indikator ekonomi utama.
Silahkan masukkan nilai antara 0-10 untuk setiap indikator.
""")

# Daftar fitur penting
important_features = ['SafetySecurity', 'Governance', 'EconomicQuality', 'LivingConditions']

# Deskripsi untuk setiap indikator
feature_descriptions = {
    'SafetySecurity': 'Tingkat Keamanan dan Keselamatan',
    'Governance': 'Tata Kelola Pemerintahan',
    'EconomicQuality': 'Kualitas Ekonomi',
    'LivingConditions': 'Kondisi Kehidupan'
}

# Definisi deskripsi untuk setiap cluster
cluster_descriptions = {
    0: "Negara dengan Ekonomi Tertinggal",
    1: "Negara dengan Ekonomi Berkembang",
    2: "Negara dengan Ekonomi Maju"
}

# Membuat dua kolom untuk tata letak
col1, col2 = st.columns([1, 2])

# Kolom input
with col1:
    st.subheader("📊 Input Indikator Ekonomi")
    
    # Form input
    with st.form(key="prediction_form"):
        input_data = {}
        
        # Membuat slider untuk setiap fitur
        for feature in important_features:
            input_data[feature] = st.slider(
                feature_descriptions[feature],
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                key=f"slider_{feature}",
                help=f"Masukkan nilai untuk {feature_descriptions[feature]} (0-10)"
            )
        
        # Tombol submit
        submit_button = st.form_submit_button("Prediksi Cluster")

# Proses prediksi
if submit_button:
    try:
        # Validasi input
        is_valid, error_message = validate_input(input_data)
        
        if not is_valid:
            st.error(error_message)
        else:
            with st.spinner('Melakukan prediksi...'):
                # Konversi input ke DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Scaling input
                scaled_input = st.session_state.scaler.transform(input_df)
                
                # Prediksi
                cluster = st.session_state.model.predict(scaled_input)[0]
                st.session_state.prediction_result = cluster
                
                # Tampilkan hasil di kolom kedua
                with col2:
                    st.subheader("🎯 Hasil Prediksi")
                    
                    # Tampilkan prediksi dengan styling
                    st.success(f"""
                        ### Cluster Hasil Prediksi: {cluster}
                        **Klasifikasi:** {cluster_descriptions[cluster]}
                    """)
                    
                    # Visualisasi input
                    st.subheader("📊 Visualisasi Input")
                    chart_data = pd.DataFrame(
                        [input_data.values()],
                        columns=[feature_descriptions[feat] for feat in input_data.keys()]
                    )
                    st.bar_chart(chart_data.T)
                    
                    # Tampilkan ringkasan input
                    st.subheader("📋 Ringkasan Input")
                    summary_df = input_df.copy()
                    summary_df.columns = [feature_descriptions[col] for col in summary_df.columns]
                    st.dataframe(summary_df)
                    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Dibuat dengan ❤️ menggunakan Streamlit | Proyek Machine Learning</p>
</div>
""", unsafe_allow_html=True)
