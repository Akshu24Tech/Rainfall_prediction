import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/random_forest.joblib")

# Title
st.title("🌧️ Rainfall Prediction App")
st.write("Enter monthly rainfall (in mm) to predict annual total")

# User inputs
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

inputs = []
for month in months:
    val = st.number_input(f"{month} Rainfall (mm)", min_value=0.0, step=0.1)
    inputs.append(val)

# Prediction button
if st.button("Predict Annual Rainfall"):
    input_array = np.array(inputs).reshape(1, -1)
    predicted = model.predict(input_array)[0]
    st.success(f"☔ Predicted Annual Rainfall: **{predicted:.2f} mm**")
