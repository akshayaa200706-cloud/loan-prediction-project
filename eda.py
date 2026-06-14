import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/archive/cleaned_loan_data.csv")
plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)

plt.title("Loan Approval Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Education', data=df)

plt.title("Education Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)

plt.title("Gender Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Property_Area', data=df)

plt.title("Property Area Distribution")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)

plt.title("Credit History vs Loan Approval")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(df['ApplicantIncome'], bins=30)

plt.title("Applicant Income Distribution")
plt.show()
print(df.select_dtypes(include=['int64','float64']).corr())
