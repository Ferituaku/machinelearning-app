import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Cluster Negara",
    page_icon="🌍",
    layout="centered"
)

# Daftar Semua Fitur
all_features = [
    'SafetySecurity', 'PersonelFreedom', 'Governance',
    'SocialCapital', 'InvestmentEnvironment', 'EnterpriseConditions',
    'MarketAccessInfrastructure', 'EconomicQuality', 'LivingConditions',
    'Health', 'Education', 'NaturalEnvironment'
]

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('kmeans_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model dan scaler
model, scaler = load_model()

if model is None or scaler is None:
    st.error("Failed to load model or scaler. Please check if files exist.")
    st.stop()

# Judul dan Deskripsi
st.title("🌍 Prediksi Cluster Negara")
st.write("""
Masukkan nilai indikator ekonomi negara berdasarkan skala 0-10 untuk mengetahui cluster ekonomi negara.
Semua indikator harus diisi untuk mendapatkan hasil prediksi yang akurat.
""")

# Form Input
with st.form("prediction_form"):
    # Dictionary untuk menyimpan input
    input_data = {}
    
    # Buat 3 kolom untuk layout yang lebih baik
    col1, col2, col3 = st.columns(3)
    
    # Distribusikan fitur ke kolom
    columns = [col1, col2, col3]
    features_per_column = len(all_features) // 3
    
    for idx, feature in enumerate(all_features):
        col_idx = idx // features_per_column
        if col_idx < len(columns):
            with columns[col_idx]:
                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=0.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.1,
                    key=f"slider_{feature}",
                    help=f"Masukkan nilai untuk {feature}"
                )
    
    # Tombol Submit
    submitted = st.form_submit_button("Prediksi Cluster")

# Proses prediksi ketika form disubmit
if submitted:
    try:
        # Buat DataFrame dari input
        input_df = pd.DataFrame([input_data])
        
        # Standarisasi input menggunakan scaler
        scaled_input = scaler.transform(input_df)
        
        # Prediksi cluster
        predicted_cluster = model.predict(scaled_input)[0]
        
        # Tampilkan hasil
        st.subheader("📋 Hasil Prediksi")
        
        # Definisi cluster
        cluster_definitions = {
            0: "Negara dengan Ekonomi Tertinggal",
            1: "Negara dengan Ekonomi Berkembang",
            2: "Negara dengan Ekonomi Maju"
        }
        
        # Tampilkan hasil dengan styling
        st.success(f"""
        ### Cluster yang Diprediksi: {predicted_cluster}
        **Klasifikasi:** {cluster_definitions[predicted_cluster]}
        """)
        
        # Tampilkan visualisasi
        st.subheader("📊 Visualisasi Input")
        
        # Bar chart
        st.bar_chart(input_df.T)
        
        # Tabel detail
        st.subheader("📋 Detail Input")
        # Format angka menjadi 2 desimal
        formatted_df = input_df.round(2)
        st.dataframe(formatted_df, use_container_width=True)
        
        # Tambahkan interpretasi
        st.subheader("💡 Interpretasi")
        st.write(f"""
        Berdasarkan input yang diberikan, negara ini memiliki karakteristik yang sesuai dengan 
        {cluster_definitions[predicted_cluster].lower()}. 
        
        Beberapa indikator utama:
        - Nilai tertinggi: {input_df.idxmax(axis=1)[0]} ({input_df.max(axis=1)[0]:.2f})
        - Nilai terendah: {input_df.idxmin(axis=1)[0]} ({input_df.min(axis=1)[0]:.2f})
        - Rata-rata nilai: {input_df.mean(axis=1)[0]:.2f}
        """)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created with ❤️ by CUKIMAK | Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)

# Tambahkan CSS untuk styling
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
    div.row-widget.stButton {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
