import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction App")

# -------------------------------------------------
# Load Model (SAFE)
# -------------------------------------------------
MODEL_PATH = "churn_prediction_model (1).pkl"

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model = None

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 10000.0)
num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"])
is_active = st.sidebar.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# -------------------------------------------------
# Prepare Input Data
# -------------------------------------------------
if st.sidebar.button("Predict Churn") and model is not None:

    input_data = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
        "Gender": 1 if gender == "Male" else 0
    }

    df = pd.DataFrame([input_data])

    # Geography Encoding (IMPORTANT)
    df["Geography_France"] = 1 if geography == "France" else 0
    df["Geography_Germany"] = 1 if geography == "Germany" else 0
    df["Geography_Spain"] = 1 if geography == "Spain" else 0

    # -------------------------------------------------
    # Align features with model
    # -------------------------------------------------
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]
        else:
            prob = model.predict(df)[0]

        prediction = "🔴 Customer Will Churn" if prob >= 0.5 else "🟢 Customer Will Stay"

        st.subheader("Prediction Result")
        st.success(prediction)
        st.info(f"Churn Probability: **{prob:.2f}**")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Churn Prediction App | Streamlit + ML")
