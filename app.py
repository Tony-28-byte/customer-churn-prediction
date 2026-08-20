import streamlit as st
import joblib
import pandas as pd


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================
# Load Model and Threshold
# =========================

model = joblib.load("customer_churn_model.pkl")
threshold = joblib.load("churn_threshold.pkl")


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("📊 About the Model")

    st.write(
        "This application predicts the probability "
        "that a customer will churn."
    )

    st.divider()

    st.write("**Model:** Random Forest")
    st.write("**Features:** 19")
    st.write("**Decision Threshold:** 30%")

    st.divider()

    st.caption(
        "Built as a customer churn prediction machine learning project."
    )


# =========================
# App Header
# =========================

st.title("📊 Customer Churn Prediction")

st.markdown(
    "Predict the likelihood of customer churn using a machine learning model."
)


# =========================
# Customer Information
# =========================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )


# =========================
# Personal Information
# =========================

st.subheader("👤 Personal Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

with col4:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )


# =========================
# Services
# =========================

st.subheader("📡 Services")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# =========================
# Contract & Payment
# =========================

st.subheader("📄 Contract & Payment")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Credit card (automatic)",
            "Bank transfer (automatic)"
        ]
    )


# =========================
# Prepare Input Data
# =========================

input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method]
})


# =========================
# Prediction
# =========================

st.divider()

st.subheader("🔮 Churn Prediction")

predict_button = st.button(
    "Predict Churn",
    use_container_width=True
)


if predict_button:

    probability = model.predict_proba(input_data)[0, 1]

    prediction = 1 if probability >= threshold else 0

    st.divider()

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.0%}"
        )

    if prediction == 1:

        st.error("⚠️ High Churn Risk")

        st.write(
        f"The model estimates a {probability:.2%} probability "
        "that this customer will churn."
    )

        st.warning(
        "Consider proactive customer retention actions."
    )

    else:

        st.success("✅ Low Churn Risk")

        st.write(
        f"The model estimates a {probability:.2%} probability "
        "that this customer will churn."
    )

        st.info(
        "The customer is currently below the selected churn threshold."
    )


st.divider()

st.caption(
    "Customer Churn Prediction • Machine Learning Project"
)


