import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests

st.title("💳 Loan Default Checker")

@st.cache_resource
def load_model():
    model_path = "loan_default_model.pkl"
    url = "https://huggingface.co/Raghss/loan-default-model/resolve/main/loan_default_model.pkl"

    if not os.path.exists(model_path):
        with st.spinner("🔽 Downloading model..."):
            r = requests.get(url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    return joblib.load(model_path)

model = load_model()

st.subheader("📋 Enter Loan Information")
loan_amnt = st.number_input("Loan Amount ($)", 500, 40000, 10000)
int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
emp_length = st.slider("Employment Length (years)", 0, 10, 5)
annual_inc = st.number_input("Annual Income ($)", 10000, 1000000, 50000)
dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)

if st.button("🔍 Check Risk"):
    input_data = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "emp_length": emp_length,
        "annual_inc": annual_inc,
        "dti": dti
    }])

    # Fill missing required model columns with zeros
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.markdown("### ❌ High Risk of Default")
        st.markdown(f"🔴 **Probability of Default: `{prob:.2%}`**")
        st.markdown('<div style="font-size:80px; color:red;">●</div>', unsafe_allow_html=True)
    else:
        st.markdown("### ✅ Low Risk – Likely to Repay")
        st.markdown(f"🟢 **Probability of Default: `{prob:.2%}`**")
        st.markdown('<div style="font-size:80px; color:green;">●</div>', unsafe_allow_html=True)
