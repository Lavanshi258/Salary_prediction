import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv("salary_data.csv")

X = df[['YearsExperience']]
y = df['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('salary_model.pkl', 'wb'))

print("Model trained successfully!")