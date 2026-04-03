import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------
# LOAD MODEL & COLUMNS
# -------------------------------
model = pickle.load(open("models/model.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

# -------------------------------
# UI TITLE
# -------------------------------
st.title("📊 YouTube Revenue Predictor")

st.markdown("Enter video performance details to estimate ad revenue")

# -------------------------------
# USER INPUTS
# -------------------------------
views = st.number_input("Views", min_value=1, value=1000)
likes = st.number_input("Likes", min_value=0, value=100)
comments = st.number_input("Comments", min_value=0, value=10)
watch_time = st.number_input("Watch Time (minutes)", min_value=1, value=500)

# -------------------------------
# SAFE FEATURE ENGINEERING
# -------------------------------
if views > 0:
    engagement = (likes + comments) / views
    watch_time_per_view = watch_time / views
else:
    engagement = 0
    watch_time_per_view = 0

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("Predict Revenue"):

    # Create input dictionary
    input_dict = {
        "views": views,
        "likes": likes,
        "comments": comments,
        "watch_time_minutes": watch_time,
        "engagement_rate": engagement,
        "watch_time_per_view": watch_time_per_view
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply same encoding as training
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.success(f"💰 Estimated Revenue: ${prediction[0]:,.2f}")

    # Debug (optional - remove later)
    # st.write(input_df.head())