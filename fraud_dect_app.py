import streamlit as st
import joblib
import numpy as np
import pandas as pd
# Load model and scaler
model = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\fraud_trans_dect.pro\fraud_detection\fraud_detection_log_reg_model.joblib")
scaler = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\fraud_trans_dect.pro\fraud_detection\scaler.joblib")

st.title("💳 Fraud Transaction Detector")
st.markdown("Enter transaction details below to check if it's fraudulent.")

# Transaction Amount
tx_amount = st.text_input("💰 Transaction Amount", help="Enter the transaction amount in dollars (e.g., 100.50)")

# Time dropdown (intervals in hours)
time_options = {f"{h:02d}:00": h * 3600 for h in range(24)}
tx_time_label = st.selectbox("🕒 Transaction Time", list(time_options.keys()), help="Select the hour when the transaction occurred")
tx_time_seconds = time_options[tx_time_label]

# Transaction Location
tx_location = st.selectbox("📍 Transaction Location ID", list(range(1, 21)), help="Select a location ID")

# Customer ID
customer_id = st.selectbox("👤 Customer ID", list(range(1000, 1010)), help="Select a customer ID")

# Terminal ID
terminal_id = st.selectbox("🏧 Terminal ID", list(range(200, 210)), help="Select a terminal ID")

# Day of Month
tx_day = st.selectbox("📅 Day of the Month", list(range(1, 32)))

# Month
months_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
month_name = st.selectbox("🗓️ Month", list(months_map.keys()))
tx_month = months_map[month_name]

# Year
tx_year = st.selectbox("📆 Year", [2022, 2023, 2024, 2025])

# Day of Week
weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
weekday_name = st.selectbox("📌 Day of the Week", list(weekday_map.keys()))
tx_weekday = weekday_map[weekday_name]

if st.button("Predict Fraud"):
    try:
        # Derived values
        tx_time_days = tx_day  # assuming 1 day = 1 unit here
        tx_hour = int(tx_time_seconds) // 3600
        tx_day_of_week = tx_weekday

        # Prepare input features
        feature_inputs = [
            float(tx_location), float(customer_id), float(terminal_id),
            float(tx_amount), float(tx_time_seconds), float(tx_time_days),
            float(tx_hour), float(tx_day_of_week)
        ]

        # Define column names in the same order used during training
        feature_names = ['TRANSACTION_ID', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT',
                         'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'TX_HOUR', 'TX_DAY_OF_WEEK']

        # Convert to DataFrame to match scaler expectations
        input_df = pd.DataFrame([feature_inputs], columns=feature_names)

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]

        # Output
        if prediction == 1:
            st.error("⚠️ This transaction is likely **FRAUDULENT**.")
        else:
            st.success("✅ This transaction seems **LEGITIMATE**.")

    except ValueError:
        st.warning("⚠️ Please fill in **all fields** with valid numbers.")
