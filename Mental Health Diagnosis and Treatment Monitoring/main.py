import streamlit as st
import numpy as np
import joblib

model = joblib.load('xgboost_model.joblib')

st.title("Prediksi Kualitas Tidur Berdasarkan Faktor Kesehatan Mental")
st.write("Aplikasi ini memprediksi kualitas tidur pasien berdasarkan berbagai faktor kesehatan mental.")

age = st.number_input("Usia:", min_value=10, max_value=100, value=25, step=1)
gender = st.radio("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
symptom_severity = st.slider("Tingkat Keparahan Gejala (1-10):", 1, 10, 5)
mood_score = st.slider("Skor Suasana Hati (1-10):", 1, 10, 5)
physical_activity = st.number_input("Aktivitas Fisik (jam/minggu):", min_value=0.0, max_value=50.0, value=3.0)
treatment_duration = st.number_input("Durasi Perawatan (minggu):", min_value=1, max_value=100, value=10, step=1)
stress_level = st.slider("Tingkat Stres (1-10):", 1, 10, 5)
treatment_progress = st.slider("Progres Perawatan (1-10):", 1, 10, 5)
adherence_to_treatment = st.slider("Kepatuhan Terapi (%):", 0, 100, 50)

if st.button("Prediksi"):
    data_input = np.array([
        age, gender, symptom_severity, mood_score, 
        physical_activity, treatment_duration, stress_level, 
        treatment_progress, adherence_to_treatment
    ]).reshape(1, -1)

    sleep_quality_prediction = model.predict(data_input)[0]
    
    if sleep_quality_prediction >= 7:
        kategori = "Baik"
    elif sleep_quality_prediction >= 4:
        kategori = "Sedang"
    else:
        kategori = "Buruk"

    st.write("### === Hasil Prediksi ===")
    st.write(f"**Perkiraan Kualitas Tidur:** {sleep_quality_prediction:.2f} dari 10")
    st.write(f"**Kategori:** {kategori}")