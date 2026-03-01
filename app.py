import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("churn_model.pkl")

# Title
st.markdown(
    "<h1 style='text-align: center;'>Customer Churn Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>AI-powered churn risk analysis for customer retention</p>",
    unsafe_allow_html=True
)

st.divider()

# Sidebar Inputs
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 80, 30)
tenure = st.sidebar.slider("Tenure (Months)", 0, 120, 12)
usage = st.sidebar.slider("Usage Frequency", 0, 100, 10)
support = st.sidebar.slider("Support Calls", 0, 20, 2)
payment_delay = st.sidebar.slider("Payment Delay (Days)", 0, 60, 5)
total_spend = st.sidebar.number_input("Total Spend", 0.0, 100000.0, 5000.0)
last_interaction = st.sidebar.slider("Days Since Last Interaction", 0, 365, 30)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
subscription = st.sidebar.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.sidebar.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

st.sidebar.divider()

predict_button = st.sidebar.button("Predict Churn")

# Main Area
if predict_button:

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage],
        "Support Calls": [support],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription],
        "Contract Length": [contract],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{probability:.2%}")

        st.progress(float(probability))

    with col2:
        if prediction == 1:
            st.error("High Risk of Churn!")
            st.markdown("### Risk Level: HIGH")
        else:
            st.success("Low Risk of Churn!")
            st.markdown("### Risk Level: LOW")

    st.divider()

    st.subheader("Customer Profile Summary")
    st.dataframe(input_data, use_container_width=True)

else:
    st.info("Enter customer details in the sidebar and click Predict Churn")
