import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("E:/Credit_scoring_model/models/random_forest_model.pkl")
scaler = joblib.load("E:/Credit_scoring_model/models/scaler.pkl")
st.set_page_config(page_title="Credit Default Predictor", page_icon="💳")
st.title("💳 Credit Default Risk Prediction")
st.markdown("Enter your values or click the **Fill Sample** button to auto-fill example values.")

# Sample Profiles
sample_values = [
    {
        "label": "👤 Sample 1 (35 yrs, avg usage)",
        "credit_usage": 0.65,
        "age": 35,
        "debt_ratio": 0.45,
        "monthly_income": 4500,
        "open_credit": 5,
        "times_late": 1,
        "real_estate": 1,
        "dependents": 2
    },
    {
        "label": "👤 Sample 2 (28 yrs, low risk)",
        "credit_usage": 0.25,
        "age": 28,
        "debt_ratio": 0.35,
        "monthly_income": 3500,
        "open_credit": 4,
        "times_late": 0,
        "real_estate": 0,
        "dependents": 1
    },
    {
        "label": "👤 Sample 3 (52 yrs, high usage)",
        "credit_usage": 0.92,
        "age": 52,
        "debt_ratio": 1.2,
        "monthly_income": 8000,
        "open_credit": 7,
        "times_late": 3,
        "real_estate": 2,
        "dependents": 3
    },
    {
        "label": "👤 Sample 4 (41 yrs, moderate risk)",
        "credit_usage": 0.48,
        "age": 41,
        "debt_ratio": 0.6,
        "monthly_income": 6000,
        "open_credit": 6,
        "times_late": 1,
        "real_estate": 1,
        "dependents": 0
    },
    {
        "label": "👤 Sample 5 (30 yrs, high util)",
        "credit_usage": 0.81,
        "age": 30,
        "debt_ratio": 0.9,
        "monthly_income": 4000,
        "open_credit": 3,
        "times_late": 2,
        "real_estate": 0,
        "dependents": 2
    },
    {
        "label": "👤 Sample 6 (65 yrs, high income)",
        "credit_usage": 0.12,
        "age": 65,
        "debt_ratio": 0.2,
        "monthly_income": 15000,
        "open_credit": 10,
        "times_late": 0,
        "real_estate": 2,
        "dependents": 1
    }
]

# Select Sample Profile (Dropdown)
sample_labels = [s["label"] for s in sample_values]
selected_sample_label = st.selectbox("🧪 Select Sample Profile (Optional)", ["None"] + sample_labels)

# Find selected sample or use manual input
selected_sample = next((s for s in sample_values if s["label"] == selected_sample_label), None)

# Form inputs
credit_usage = st.slider("Credit Usage Ratio (%)", 0.0, 1.0,
                         selected_sample["credit_usage"] if selected_sample else 0.3)
age = st.number_input("Age (Years)", 18, 100,
                      selected_sample["age"] if selected_sample else 30)
debt_ratio = st.number_input("Debt-to-Income Ratio", 0.0, 5.0,
                             selected_sample["debt_ratio"] if selected_sample else 0.5)
monthly_income = st.number_input("Monthly Income (₹)", 0.0, 1e6,
                                 float(selected_sample["monthly_income"]) if selected_sample else 4000.0)
open_credit = st.number_input("Open Credit Accounts", 0, 50,
                              selected_sample["open_credit"] if selected_sample else 3)
times_late = st.number_input("Times 90+ Days Late", 0, 100,
                             selected_sample["times_late"] if selected_sample else 0)
real_estate = st.number_input("Real Estate Loans", 0, 20,
                              selected_sample["real_estate"] if selected_sample else 1)
dependents = st.number_input("Number of Dependents", 0, 20,
                             selected_sample["dependents"] if selected_sample else 1)


# Predict button
if st.button("🔍 Predict Default Risk"):
    monthly_income_log = np.log1p(monthly_income)
    high_util_flag = 1 if credit_usage > 0.8 else 0
    debt_income_interaction = debt_ratio * monthly_income

    final_input = np.array([[credit_usage, age, debt_ratio, monthly_income,
                             open_credit, times_late, real_estate, dependents,
                             monthly_income_log, high_util_flag, debt_income_interaction]])

    scaled_input = scaler.transform(final_input)
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"❌ High Default Risk! Probability: {proba:.2%}")
    else:
        st.success(f"✅ No Default Risk. Probability: {proba:.2%}")
