import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

# App title
st.title("Diabetes Prediction System")

# User inputs
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Prediction button
if st.button("Predict"):
    data = pd.DataFrame({
            "Pregnancies": [preg],
                    "Glucose": [glucose],
                            "BloodPressure": [bp],
                                    "SkinThickness": [skin],
                                            "Insulin": [insulin],
                                                    "BMI": [bmi],
                                                            "DiabetesPedigreeFunction": [dpf],
                                                                    "Age": [age]
                                                                        })
prediction = model.predict(data)[0]
probability = model.predict_proba(data)[0][1]

if prediction == 1:
        st.error(f"Diabetic Risk Detected ({probability:.2%})")
else:
       st.success(f"No Diabetes Risk ({1 - probability:.2%})")