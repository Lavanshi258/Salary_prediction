import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("loan_data.csv")

df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df['Married'] = df['Married'].map({'Yes':1, 'No':0})
df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})

X = df[['Gender','Married','Education','Self_Employed',
        'ApplicantIncome','CoapplicantIncome',
        'LoanAmount','Loan_Amount_Term','Credit_History']]

y = df['Loan_Status'].map({'Y':1,'N':0})

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open('loan_model.pkl', 'wb'))

print("Model trained and saved!")