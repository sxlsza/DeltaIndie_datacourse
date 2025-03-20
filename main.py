import streamlit as st
import numpy as np
import joblib

model = joblib.load('xgboost_model.joblib')

st.title("Prediksi Kepatuhan Perawatan Kesehatan Mental Berdasarkan Kualitas Tidur dan Faktor Lainnya")
st.write("Aplikasi ini memprediksi tingkat kepatuhan pasien terhadap perawatan berdasarkan berbagai faktor kesehatan mental.")

age = st.number_input("Usia:", min_value=10, max_value=100, value=25, step=1)
gender = st.radio("Jenis Kelamin", options=[0,1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
symptom_severity = st.slider("Tingkat Keparahan Gejala (1-10):", 1, 10, 5)
mood_score = st.slider("Skor Suasana Hati (1-10):", 1, 10, 5)
sleep_quality = st.slider("Kualitas Tidur (1-10):", 1, 10, 5)
physical_activity = st.number_input("Aktivitas Fisik (jam/minggu):", min_value=0.0, max_value=50.0, value=3.0)
treatment_duration = st.number_input("Durasi Perawatan (minggu):", min_value=1, max_value=100, value=10, step=1)
stress_level = st.slider("Tingkat Stres (1-10):", 1, 10, 5)
treatment_progress = st.slider("Progres Perawatan (1-10):", 1, 10, 5)

if st.button("Prediksi"):
    data_input = np.array([
        age, gender, symptom_severity, mood_score, sleep_quality,
        physical_activity, treatment_duration, stress_level,
        treatment_progress
    ]).reshape(1, -1)

    adherence_prediction = model.predict(data_input)[0]
    st.success(f"Prediksi Kepatuhan terhadap Perawatan: {adherence_prediction:.2f}%")
    