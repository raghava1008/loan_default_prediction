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
        with st.spinner("Downloading model..."):
            r = requests.get(model_url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    model = joblib.load(model_path)
    return model

model = load_model()

with st.form("predict_form"):
    loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 10000)
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0)
    emp_length = st.slider("Employment Length (years)", 0, 10, 5)
    annual_inc = st.number_input("Annual Income ($)", 10000, 1000000, 60000)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
    fico_low = st.slider("FICO Score (Low)", 600, 850, 690)
    fico_high = st.slider("FICO Score (High)", 600, 850, 705)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_dict = {
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'emp_length': emp_length,
        'annual_inc': annual_inc,
        'dti': dti,
        'fico_range_low': fico_low,
        'fico_range_high': fico_high,
    }

    # Fill remaining required features with 0
    all_features = list(model.feature_names_in_)
    for feat in all_features:
        if feat not in input_dict:
            input_dict[feat] = 0

    input_df = pd.DataFrame([input_dict])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.markdown("### ❌ High Risk of Default")
        st.markdown(f"🔴 **Probability: `{prob:.2%}`**")
        st.markdown('<h1 style="font-size:80px; color:red;">●</h1>', unsafe_allow_html=True)
    else:
        st.markdown("### ✅ Likely to Repay")
        st.markdown(f"🟢 **Probability of Default: `{prob:.2%}`**")
        st.markdown('<h1 style="font-size:80px; color:green;">●</h1>', unsafe_allow_html=True)
