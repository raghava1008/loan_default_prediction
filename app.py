import streamlit as st
import pandas as pd
import joblib
import os
import requests

st.title("💳 Loan Default Risk Predictor")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = "loan_default_model.pkl"
    model_url = "https://huggingface.co/Raghss/loan-default-model/resolve/main/loan_default_model.pkl"

    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model..."):
            r = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)

    return joblib.load(model_path)

model = load_model()

# --- User Inputs ---
st.subheader("Enter Borrower Information:")

loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=15000)
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0)
annual_inc = st.number_input("Annual Income ($)", 10000, 1000000, 60000)
dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
emp_length = st.slider("Employment Length (years)", 0, 10, 5)

if st.button("🔍 Predict"):
    input_dict = {
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'annual_inc': annual_inc,
        'dti': dti,
        'emp_length': emp_length
    }

    input_df = pd.DataFrame([input_dict])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"❌ High Risk of Default - Probability: {prob:.2%}")
        st.markdown("<h1 style='color:red;'>🔴</h1>", unsafe_allow_html=True)
    else:
        st.success(f"✅ Likely to Repay - Probability of Default: {prob:.2%}")
        st.markdown("<h1 style='color:green;'>🟢</h1>", unsafe_allow_html=True)
