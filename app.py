import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests

st.title("💳 Loan Default Predictor")
st.markdown("Enter borrower details and predict loan default risk.")

@st.cache_resource
def load_model():
    model_path = "loan_default_model.pkl"
    model_url = "https://huggingface.co/Raghss/loan-default-model/resolve/main/loan_default_model.pkl"

    if not os.path.exists(model_path):
        try:
            st.info("📥 Downloading model from Hugging Face...")
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# --- Input Form ---
with st.form("predict_form"):
    st.subheader("📝 Borrower Information")
    loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=40000, value=10000)
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0)
    emp_length = st.slider("Employment Length (years)", 0, 10, 5)
    annual_inc = st.number_input("Annual Income", 10000, 1000000, value=50000)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
    fico_low = st.slider("FICO Score (Low Range)", 600, 850, 690)
    fico_high = st.slider("FICO Score (High Range)", 600, 850, 705)

    submit = st.form_submit_button("🔍 Predict Default Risk")

# --- Predict and Show ---
if submit:
    input_data = {
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'emp_length': emp_length,
        'annual_inc': annual_inc,
        'dti': dti,
        'fico_range_low': fico_low,
        'fico_range_high': fico_high,
    }

    df = pd.DataFrame([input_data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    if pred == 1:
        st.error(f"❌ High Risk of Default\n\n**Probability: {prob:.2%}**")
        st.markdown('<h1 style="color:red; font-size:60px;">🔴</h1>', unsafe_allow_html=True)
    else:
        st.success(f"✅ Likely to Repay\n\n**Probability of Default: {prob:.2%}**")
        st.markdown('<h1 style="color:green; font-size:60px;">🟢</h1>', unsafe_allow_html=True)
