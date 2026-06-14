import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/archive/cleaned_loan_data.csv")

print(df.head())
df = df.drop("Loan_ID", axis=1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
df["Dependents"] = le.fit_transform(df["Dependents"])
df["Education"] = le.fit_transform(df["Education"])
df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])
print(df.head())
print(df.info())
df.to_csv("data/archive/encoded_loan_data.csv", index=False)

print("Encoded dataset saved successfully!")