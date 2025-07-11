import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.title("💳 Loan Default Predictor")
st.markdown("Upload your trained `.pkl` model and enter borrower details to predict default risk.")

# --- Upload model ---
st.sidebar.header("📤 Upload Model")
uploaded_model = st.sidebar.file_uploader("Upload a .pkl model file", type=["pkl"])

if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
else:
    st.warning("⚠️ Please upload a trained model to continue.")
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

# --- Prediction ---
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
    try:
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        if pred == 1:
            st.error(f"❌ High Risk of Default\n\n**Probability: {prob:.2%}**")
            st.markdown('<h1 style="color:red; font-size:60px;">🔴</h1>', unsafe_allow_html=True)
        else:
            st.success(f"✅ Likely to Repay\n\n**Probability of Default: {prob:.2%}**")
            st.markdown('<h1 style="color:green; font-size:60px;">🟢</h1>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
