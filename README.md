# 💳 Loan Default Prediction App

This is an interactive **Streamlit web application** that predicts whether a borrower is likely to default on a loan. Built using a trained Random Forest model on LendingClub-style loan data.

![App Logo](logo.png)

---

## 🚀 Features

- ✅ **Single Loan Prediction** via user input
- 📂 **Batch Prediction** via CSV upload
- 🔵 **Default Probability Estimate**
- 📊 **Top 5 Important Features** shown per prediction
- 🟢 Red/Green visual indicator for default risk
- 📥 **Downloadable Predictions CSV**
- 📈 **Actual vs Predicted Chart** for batch mode
- 🎨 Clean, custom-styled UI with logo

---

## 🧠 Model Info

The model is a `RandomForestClassifier` trained on:
- Loan amount, interest rate, DTI, FICO scores, employment length, etc.
- Categorical features are one-hot encoded.
- The target is whether a loan was **defaulted or charged-off**.

You can find the training script in [`train.py`](train.py).

---

## 🗂 Project Structure

