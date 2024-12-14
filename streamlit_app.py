import streamlit as st
import pandas as pd
st.title('Project Aplikasi Model Prediksi')

st.write('Hai, Selamat Datang!')

with st.expander('Data'):
  st.write('**Raw Data')
  df = pd.read_csv("https://raw.githubusercontent.com/Ferituaku/machinelearning-app/refs/heads/master/data.csv")
  df
