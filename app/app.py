import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/loan_model.pkl")

st.title("🏦 Loan Approval Prediction System")

st.markdown("""
Predict whether a loan application
will be approved based on applicant details.
""")

st.write("Enter applicant details below")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox(
    "Loan Term (Months)",
    [360, 180, 120, 84, 60,]
)

credit_history = st.selectbox(
    "Credit History",
    ["Good", "Poor"]
)
property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

if st.button("Predict"):
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Good" else 0

    dependents_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3+": 3
    }

    property_map = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }

    input_data = pd.DataFrame([[
        gender,
        married,
        dependents_map[dependents],
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_map[property_area]
    ]])
    st.write("Input Data:")
    st.write(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric(
        "Approval Probability",
        f"{probability[0][1]*100:.2f}%"
    )
    st.markdown("---")

st.caption(
    "Developed by Akshayaa | Machine Learning Project"
)