import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('salary_model.pkl', 'rb'))

st.title("Salary Prediction App 💼")

experience = st.number_input("Enter Years of Experience", min_value=0.0)

if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    st.success(f"Predicted Salary: ₹ {int(prediction[0])}")