import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Trained Model
# ----------------------------
model = joblib.load("loan_approval_prediction.pkl")

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant details below to predict whether the loan will be approved.")

# ----------------------------
# User Inputs
# ----------------------------

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    [0, 1],
    help="0 = No, 1 = Yes"
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=1000000
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1,
    value=12
)

cibil_score = st.slider(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=0
)

# ----------------------------
# One-Hot Encoding
# ----------------------------

education_graduate = 1 if education == "Graduate" else 0
education_not_graduate = 1 if education == "Not Graduate" else 0

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "self_employed": [self_employed],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value],
        "education_Graduate": [education_graduate],
        "education_Not Graduate": [education_not_graduate]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.write(f"Approval Probability: **{probability[1]*100:.2f}%**")
    else:
        st.error("❌ Loan Rejected")
        st.write(f"Rejection Probability: **{probability[0]*100:.2f}%**")

st.markdown("---")
st.caption("Developed using Streamlit & Scikit-Learn")