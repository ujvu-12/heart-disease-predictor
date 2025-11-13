# app.py
# 💖 Streamlit Web App for Heart Disease Prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open("models/heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="centered")

# App Header
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Sidebar user input
st.sidebar.header("Input Patient Data")

def user_input():
    age = st.sidebar.slider("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.sidebar.selectbox("Resting ECG Results", ["normal", "ST-T abnormality", "left ventricular hypertrophy"])
    thalch = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["False", "True"])
    oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    ca = st.sidebar.slider("Number of Major Vessels (0–3)", 0, 3, 0)
    thal = st.sidebar.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    # Encode categorical values same as in training
    sex = 1 if sex == "Male" else 0
    cp_map = {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3}
    cp = cp_map[cp]
    fbs = 1 if fbs == "True" else 0
    exang = 1 if exang == "True" else 0
    slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
    slope = slope_map[slope]
    thal_map = {"normal": 1, "fixed defect": 2, "reversable defect": 3}
    thal = thal_map[thal]
    restecg_map = {"normal": 0, "ST-T abnormality": 1, "left ventricular hypertrophy": 2}
    restecg = restecg_map[restecg]

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    return pd.DataFrame([data])

input_df = user_input()

# Display user inputs
st.subheader("🧾 Patient Details")
st.write(input_df)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)
    result = "💔 Likely to have Heart Disease" if prediction[0] == 1 else "💚 Not likely to have Heart Disease"
    st.subheader("Prediction Result:")
    st.success(result)
