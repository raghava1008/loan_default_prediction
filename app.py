import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64
import os
import requests

st.markdown("✅ App loaded successfully - top of file")

# Custom CSS styling
st.markdown("""
<style>
    .main { background-color: #f5f7fa; padding: 2rem; }
    h1 { color: #003366; font-size: 3em; font-weight: bold; }
    h3, h4 { color: #1a237e; }
    .circle { font-size: 60px; line-height: 0.5; }
    .circle.red { color: #ff1744; }
    .circle.green { color: #00c853; }
    .stDataFrame {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0px 1px 3px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0d47a1;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = "loan_default_model.pkl"
    model_url = "https://huggingface.co/Raghss/loan-default-model/resolve/main/loan_default_model.pkl"

    if not os.path.exists(model_path):
        with st.spinner("🌐 Downloading model from Hugging Face..."):
            try:
                r = requests.get(model_url, stream=True)
                r.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("📥 Model downloaded successfully.")
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                return None

    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


st.markdown("🧠 Starting model download from Hugging Face...")
model = load_model()
if model is None:
    st.stop()

st.image("logo.png", width=120)
st.title("💳 Loan Default Prediction App")
st.markdown("### Predict borrower risk & explore model insights")
st.markdown("---")

st.sidebar.markdown("📂 **Upload CSV for batch predictions**")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is None:
    st.subheader("📝 Enter Borrower Info")
    with st.form("loan_form"):
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000)
        term = st.selectbox("Loan Term", ['36 months', '60 months'])
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0)
        installment = st.number_input("Monthly Installment ($)", 50.0, 2000.0, 350.0)
        grade = st.selectbox("Grade", list("ABCDEFG"))
        emp_length = st.slider("Employment Length (years)", 0, 10, 5)
        home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
        annual_inc = st.number_input("Annual Income ($)", 10000, 1000000, 60000)
        verification_status = st.selectbox("Verification Status", ['Verified', 'Source Verified', 'Not Verified'])
        purpose = st.selectbox("Purpose", ['credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase', 'other'])
        dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
        delinq_2yrs = st.number_input("Delinquencies (last 2 yrs)", 0, 10, 0)
        fico_low = st.slider("FICO Score (Low Range)", 600, 850, 690)
        fico_high = st.slider("FICO Score (High Range)", 600, 850, 705)
        open_acc = st.number_input("Open Accounts", 0, 50, 10)
        pub_rec = st.number_input("Public Records", 0, 10, 0)
        revol_bal = st.number_input("Revolving Balance ($)", 0, 500000, 15000)
        revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, 55.0)
        total_acc = st.number_input("Total Accounts", 0, 100, 30)
        application_type = st.selectbox("Application Type", ['Individual', 'Joint'])

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        input_dict = {
            'loan_amnt': loan_amnt, 'int_rate': int_rate, 'installment': installment,
            'emp_length': emp_length, 'annual_inc': annual_inc, 'dti': dti,
            'delinq_2yrs': delinq_2yrs, 'fico_range_low': fico_low,
            'fico_range_high': fico_high, 'open_acc': open_acc, 'pub_rec': pub_rec,
            'revol_bal': revol_bal, 'revol_util': revol_util, 'total_acc': total_acc,
            'term_60 months': int(term == '60 months'),
            'grade_B': int(grade == 'B'), 'grade_C': int(grade == 'C'),
            'grade_D': int(grade == 'D'), 'grade_E': int(grade == 'E'),
            'grade_F': int(grade == 'F'), 'grade_G': int(grade == 'G'),
            'home_ownership_MORTGAGE': int(home_ownership == 'MORTGAGE'),
            'home_ownership_OWN': int(home_ownership == 'OWN'),
            'home_ownership_RENT': int(home_ownership == 'RENT'),
            'verification_status_Source Verified': int(verification_status == 'Source Verified'),
            'verification_status_Verified': int(verification_status == 'Verified'),
            'purpose_credit_card': int(purpose == 'credit_card'),
            'purpose_debt_consolidation': int(purpose == 'debt_consolidation'),
            'purpose_home_improvement': int(purpose == 'home_improvement'),
            'purpose_major_purchase': int(purpose == 'major_purchase'),
            'purpose_other': int(purpose == 'other'),
            'application_type_Joint': int(application_type == 'Joint')
        }

        input_df = pd.DataFrame([input_dict])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.markdown("### ❌ High Risk of Default")
            st.markdown(f"**🔴 Probability: `{prob:.2%}`**")
            st.markdown('<div class="circle red">●</div>', unsafe_allow_html=True)
        else:
            st.markdown("### ✅ Likely to Repay")
            st.markdown(f"**🟢 Probability of Default: `{prob:.2%}`**")
            st.markdown('<div class="circle green">●</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Top 5 Important Features")
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'feature': input_df.columns, 'importance': importances}).sort_values(by='importance', ascending=False).head(5)
        st.dataframe(fi_df.reset_index(drop=True))
        st.bar_chart(fi_df.set_index('feature'))

else:
    df_batch = pd.read_csv(uploaded_file)
    df_batch.fillna(0, inplace=True)

    required_cols = model.feature_names_in_
    if not all(col in df_batch.columns for col in required_cols):
        st.error("🚫 Uploaded CSV is missing required columns. Please check the input format.")
    else:
        preds = model.predict(df_batch)
        probs = model.predict_proba(df_batch)[:, 1]

        df_batch['prediction'] = preds
        df_batch['default_probability'] = probs

        st.subheader("📋 Prediction Results")
