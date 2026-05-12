# =====================================
# RESTAURANT SALES PREDICTION APP
# =====================================

import streamlit as st
import numpy as np
import pickle

# =====================================
# LOAD MODEL
# =====================================

model = pickle.load(open("model.pkl", "rb"))

# =====================================
# LOAD LABEL ENCODER
# =====================================

le = pickle.load(open("label_encoder.pkl", "rb"))

# =====================================
# PAGE TITLE
# =====================================

st.set_page_config(
    page_title="Restaurant Sales Forecasting",
    page_icon="🍽️"
)

st.title("🍽️ Restaurant Sales Prediction")

st.write("Predict restaurant sales using Machine Learning")

st.markdown("---")

# =====================================
# USER INPUTS
# =====================================

day = st.slider(
    "Select Day",
    1,
    31,
    15
)

month = st.slider(
    "Select Month",
    1,
    12,
    6
)

year = st.number_input(
    "Enter Year",
    min_value=2024,
    max_value=2035,
    value=2026
)

weekday = st.selectbox(
    "Select Weekday",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

price = st.number_input(
    "Enter Item Price",
    min_value=1.0,
    value=100.0
)

# =====================================
# FEATURE ENGINEERING
# =====================================

quarter = (month - 1) // 3 + 1

# =====================================
# LABEL ENCODING
# =====================================

day_of_week_encoded = le.transform([weekday])[0]

# =====================================
# CREATE INPUT ARRAY
# =====================================

input_data = np.array([[
    day,
    month,
    year,
    day_of_week_encoded,
    quarter,
    price
]])

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Sales"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Item Count: {prediction[0]:.2f}"
    )

    st.balloons()

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.write("Developed using Streamlit & Machine Learning")