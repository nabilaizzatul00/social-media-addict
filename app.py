aimport streamlit as st
import pandas as pd
import joblib

st.title("Prediksi Kecanduan Media Sosial 🎯")

model = joblib.load("model_pipeline.pkl")

# Input form
age = st.number_input("Umur", 10, 30, 20)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
academic_level = st.selectbox("Tingkat Pendidikan", ["High School", "Undergraduate", "Graduate"])
usage = st.slider("Rata-rata Penggunaan Sosmed (dalam jam) / Hari", 0.0, 12.0, 4.0)
platform = st.selectbox("Platform Favorit", ["Facebook", "Instagram", "Twitter", "YouTube", "TikTok"])
academic_effect = st.selectbox("Menurut Anda, apakah penggunaan sosial media anda mempengaruhi akademik?", ["Yes", "No"])
sleep = st.slider("Rata-rata total waktu tidur (dalam jam) / Hari", 0.0, 12.0, 6.0)
mental_score = st.slider("Skor Kesehatan Mental", 1, 10, 5)
relationship = st.selectbox("Status Hubungan", ["Single", "In Relationship", "Complicated"])
conflict = st.slider("Konflik karena Sosmed", 0, 10, 2)

input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Academic_Level": academic_level,
    "Avg_Daily_Usage_Hours": usage,
    "Most_Used_Platform": platform,
    "Affects_Academic_Performance": academic_effect,
    "Sleep_Hours_Per_Night": sleep,
    "Mental_Health_Score": mental_score,
    "Relationship_Status": relationship,
    "Conflicts_Over_Social_Media": conflict
}])

for col in ["Mental_Health_Score", "Conflicts_Over_Social_Media"]:
    input_data[col] = pd.cut(input_data[col], bins=[-1, 3, 6, 10], labels=["Low", "Medium", "High"])

if st.button("Prediksi"):
    prediction = model.predict(input_data)[0]
    st.success(f"Skor Kecanduan Diprediksi: {prediction:.2f}")
