import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Cluster Ekonomi Global",
    page_icon="🌍",
    layout="wide"
)

# Inisialisasi session state jika belum ada
# Ini penting untuk menyimpan model dan scaler agar tidak di-load ulang setiap kali halaman refresh
if 'model' not in st.session_state:
    try:
        st.session_state.model = pickle.load(open('kmeans_model.pkl', 'rb'))
        st.session_state.scaler = pickle.load(open('scaler.pkl', 'rb'))
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan file kmeans_model.pkl dan scaler.pkl berada dalam direktori yang sama.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        st.stop()

# Judul dan deskripsi aplikasi
st.title("🌍 Prediksi Cluster Ekonomi Negara")
st.markdown("""
Aplikasi ini memprediksi cluster ekonomi suatu negara berdasarkan indikator-indikator ekonomi utama.
Silahkan masukkan nilai antara 0-10 untuk setiap indikator.
""")

# Daftar fitur penting yang digunakan
important_features = ['SafetySecurity', 'Governance', 'EconomicQuality', 'LivingConditions']

# Membuat dua kolom untuk tata letak
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Input Indikator Ekonomi")
    
    # Membuat form input
    with st.form("form_prediksi"):
        input_data = {}
        
        # Deskripsi untuk setiap indikator
        feature_descriptions = {
            'SafetySecurity': 'Tingkat Keamanan dan Keselamatan',
            'Governance': 'Tata Kelola Pemerintahan',
            'EconomicQuality': 'Kualitas Ekonomi',
            'LivingConditions': 'Kondisi Kehidupan'
        }
        
        # Membuat slider untuk setiap fitur
        for feature in important_features:
            input_data[feature] = st.slider(
                f"{feature_descriptions[feature]}",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help=f"Masukkan nilai untuk {feature_descriptions[feature]} (0-10)"
            )
        
        # Tombol submit
        tombol_submit = st.form_submit_button("Prediksi Cluster")

# Proses prediksi ketika tombol submit ditekan
if tombol_submit:
    try:
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scaling input menggunakan scaler yang sudah dilatih
        scaled_input = st.session_state.scaler.transform(input_df)
        
        # Melakukan prediksi
        cluster = st.session_state.model.predict(scaled_input)[0]
        
        # Menampilkan hasil di kolom kedua
        with col2:
            st.subheader("🎯 Hasil Prediksi")
            
            # Definisi deskripsi untuk setiap cluster
            deskripsi_cluster = {
                0: "Negara dengan Ekonomi Tertinggal",
                1: "Negara dengan Ekonomi Berkembang",
                2: "Negara dengan Ekonomi Maju"
            }
            
            # Menampilkan prediksi dengan styling
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                <h3>Cluster Hasil Prediksi: {cluster}</h3>
                <p><strong>Klasifikasi:</strong> {deskripsi_cluster[cluster]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Membuat grafik radar menggunakan plotly
            fig = px.line_polar(
                r=[input_data[feat] for feat in important_features],
                theta=[feature_descriptions[feat] for feat in important_features],
                line_close=True,
                title="Grafik Radar Indikator Ekonomi"
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                )
            )
            st.plotly_chart(fig)
            
            # Menampilkan ringkasan input dalam tabel
            st.subheader("📋 Ringkasan Input")
            # Mengubah nama kolom untuk tampilan
            display_df = input_df.copy()
            display_df.columns = [feature_descriptions[col] for col in display_df.columns]
            st.dataframe(display_df)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")

# Menambahkan footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Dibuat dengan ❤️ oleh [Nama Anda] | Proyek Machine Learning</p>
</div>
""", unsafe_allow_html=True)
