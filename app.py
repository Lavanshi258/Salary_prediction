import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('loan_model.pkl', 'rb'))

st.title("Loan Eligibility Predictor")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

app_income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
credit_history = st.selectbox("Credit History", [1, 0])

gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if st.button("Check Eligibility"):
    data = np.array([[gender, married, education, self_employed,
                      app_income, co_income, loan_amount,
                      loan_term, credit_history]])

    result = model.predict(data)

    if result[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")