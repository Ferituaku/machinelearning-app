import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.title('Project Aplikasi Model Prediksi')

st.write('Hai, Selamat Datang!')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("https://raw.githubusercontent.com/Ferituaku/machinelearning-app/refs/heads/master/data.csv")
  df


# Load KMeans model
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))

# Definisikan semua fitur dan fitur penting
# all_features = [
#     'SafetySecurity', 'PersonelFreedom', 'Governance',
#     'SocialCapital', 'InvestmentEnvironment', 'EnterpriseConditions',
#     'MarketAccessInfrastructure', 'EconomicQuality', 'LivingConditions',
#     'Health', 'Education', 'NaturalEnvironment'
# ]
all_features = ['SafetySecurity', 'Governance', 'EconomicQuality', 'LivingConditions']

# Streamlit untuk antarmuka pengguna
st.title("Prediksi Cluster Negara")
st.write("Masukkan nilai-nilai indikator ekonomi negara untuk mengetahui clusternya.")

# Input data pengguna
input_data = {feature: 0.0 for feature in important_features}
for feature in important_features:
    input_data[feature] = st.slider(
        feature, min_value=0.0, max_value=10.0, step=0.1, value=5.0,
        help=f"Masukkan nilai untuk {feature} (0-10)"
    )

# Buat DataFrame input pengguna
input_df = pd.DataFrame([input_data])

# Standarisasi input
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_df)

# Prediksi cluster
predicted_cluster = kmeans_model.predict(scaled_input)[0]

# Tampilkan hasil prediksi
st.subheader("Hasil Prediksi")
st.write(f"Negara yang dimasukkan termasuk dalam **Cluster {predicted_cluster}**.")
st.write("""
Cluster ini mengindikasikan karakteristik ekonomi negara berdasarkan data indikator utama:
- Cluster 0: Negara ekonomi tertinggal
- Cluster 1: Negara ekonomi berkembang
- Cluster 2: Negara ekonomi maju
""")
