import pandas as pd

df = pd.read_csv("data/archive/encoded_loan_data.csv")

print(df.head())
X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]

print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!")
predictions = model.predict(X_test)

print(predictions[:10])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))
import joblib

joblib.dump(model, "models/loan_model.pkl")

print("Model saved successfully!")