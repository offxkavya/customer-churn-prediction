import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
model = joblib.load("churn_model.pkl")

# ---------------------------------------------------
# Header Section
# ---------------------------------------------------
st.markdown("<div class='main-title'>📊 Customer Churn Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered churn risk analysis for customer retention</div>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------
st.sidebar.header("📝 Enter Customer Details")

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

predict_button = st.sidebar.button("🚀 Predict Churn")

# ---------------------------------------------------
# Main Section
# ---------------------------------------------------
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

    # ---------------------------------------------------
    # Tabs
    # ---------------------------------------------------
    tab1, tab2 = st.tabs(["📈 Prediction", "📊 Analytics"])

    # ==========================
    # TAB 1 - Prediction
    # ==========================
    with tab1:

        st.subheader("📊 Churn Risk Score")

        # Gauge Chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if probability > 0.5 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},
                    {'range': [40, 70], 'color': "#f1c40f"},
                    {'range': [70, 100], 'color': "#e74c3c"},
                ],
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        # Risk Level Display
        if prediction == 1:
            st.error("⚠️ High Risk of Churn")
            st.markdown("### 🔴 Risk Level: HIGH")
        else:
            st.success("✅ Low Risk of Churn")
            st.markdown("### 🟢 Risk Level: LOW")

        # Risk Interpretation
        st.subheader("🧠 Risk Interpretation")

        if probability < 0.3:
            st.success("Customer is stable. Very low churn probability.")
        elif probability < 0.7:
            st.warning("Customer shows moderate churn signals. Monitor closely.")
        else:
            st.error("Customer is highly likely to churn. Immediate retention action recommended.")

    # ==========================
    # TAB 2 - Analytics
    # ==========================
    with tab2:

        st.subheader("📋 Customer Profile Summary")
        st.dataframe(input_data, use_container_width=True)

        st.subheader("📊 Customer Metrics Overview")

        numeric_data = {
            "Age": age,
            "Tenure": tenure,
            "Usage Frequency": usage,
            "Support Calls": support,
            "Payment Delay": payment_delay,
            "Total Spend": total_spend,
            "Last Interaction": last_interaction
        }

        chart_df = pd.DataFrame({
            "Feature": list(numeric_data.keys()),
            "Value": list(numeric_data.values())
        })

        bar_chart = go.Figure()

        bar_chart.add_trace(go.Bar(
            x=chart_df["Feature"],
            y=chart_df["Value"],
        ))

        bar_chart.update_layout(
            title="Customer Feature Distribution",
            xaxis_title="Feature",
            yaxis_title="Value",
        )

        st.plotly_chart(bar_chart, use_container_width=True)

else:
    st.info("👈 Enter customer details in the sidebar and click Predict Churn")