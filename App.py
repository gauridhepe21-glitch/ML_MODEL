
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")
encoders = joblib.load("encoder.pkl")

st.title("🫀Heart Disease Prediction")

age = st.number_input("Age",20,100)
sex = st.selectbox("Sex",["M","F"])
chest = st.selectbox("Chest Pain Type",["ATA","NAP","ASY","TA"])
bp = st.number_input("Resting BP",80,220)
chol = st.number_input("Cholesterol",0,700)
fbs = st.selectbox("Fasting Blood Sugar",[0,1])
ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
hr = st.number_input("Maximum Heart Rate",60,220)
angina = st.selectbox("Exercise Angina",["Y","N"])
oldpeak = st.number_input("Old Peak",0.0,6.0)
slope = st.selectbox("ST Slope",["Up","Flat","Down"])

if st.button("Predict"):

    data = pd.DataFrame({
        "Age":[age],
        "Sex":[sex],
        "ChestPainType":[chest],
        "RestingBP":[bp],
        "Cholesterol":[chol],
        "FastingBS":[fbs],
        "RestingECG":[ecg],
        "MaxHR":[hr],
        "ExerciseAngina":[angina],
        "Oldpeak":[oldpeak],
        "ST_Slope":[slope]
    })

    for col in encoders:
        data[col] = encoders[col].transform(data[col])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Heart Disease : YES")
    else:
        st.success("Heart Disease : NO")