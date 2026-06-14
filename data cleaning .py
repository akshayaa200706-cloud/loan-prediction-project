import pandas as pd

# Load dataset
df = pd.read_csv("data/archive/train_u6lujuX_CVtuZ9i.csv")

# Show missing values before cleaning
print("Missing Values Before Cleaning:")
print(df.isnull().sum())
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

df["Married"] = df["Married"].fillna(df["Married"].mode()[0])

df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])

df["Self_Employed"] = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())

df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mean()
)

df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mean()
)
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
df.to_csv("data/archive/cleaned_loan_data.csv", index=False)

print("\nCleaned dataset saved successfully!")