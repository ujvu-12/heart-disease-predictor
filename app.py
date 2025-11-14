# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="centered")

# -----------------------------------------------------------
# Load model, scaler and columns
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
    with open(model_path, "rb") as f:
        model, scaler, columns = pickle.load(f)
    return model, scaler, columns

model, scaler, columns = load_model()

# -----------------------------------------------------------
# Sidebar Inputs
# -----------------------------------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the patient details to check heart disease risk.")

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 20, 100, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", 
    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.sidebar.number_input("Resting BP (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])
restecg = st.sidebar.selectbox("Resting ECG", 
    ["normal", "st-t wave abnormality", "left ventricular hypertrophy"])
thalach = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [True, False])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope", ["upsloping", "flat", "downsloping"])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal", ["normal", "fixed defect", "reversable defect"])

# -----------------------------------------------------------
# Create the input dataframe
# -----------------------------------------------------------
input_dict = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}

input_df = pd.DataFrame([input_dict])

# -----------------------------------------------------------
# One-hot encode the input
# -----------------------------------------------------------
input_encoded = pd.get_dummies(input_df)

# Align with training columns (missing columns get 0)
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# Scale numeric data
scaled_input = scaler.transform(input_encoded)

# -----------------------------------------------------------
# Prediction
# -----------------------------------------------------------
if st.button("🔎 Predict"):
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("💔 High chance of Heart Disease")
    else:
        st.success("❤️ No Heart Disease Detected")

st.markdown("---")
st.caption("Made with ❤️ by Ujvwala Reddy — GitHub: ujvu-12")
