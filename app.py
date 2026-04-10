import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    "YearsExperience": [1,2,3,4,5,6,7,8,9,10],
    "Salary": [30000,35000,40000,45000,50000,55000,60000,65000,70000,75000]
}

df = pd.DataFrame(data)

X = df[['YearsExperience']]
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("Salary Prediction App 💼")

exp = st.number_input("Enter Years of Experience", min_value=0.0)

if st.button("Predict Salary"):
    prediction = model.predict([[exp]])
    st.success(f"Predicted Salary: ₹ {int(prediction[0])}")