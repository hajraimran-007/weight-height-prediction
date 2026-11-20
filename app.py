
import streamlit as st
import pandas as pd
import requests

# Page Configuration
st.set_page_config(
    page_title="Weight to Height Prediction",
    page_icon="📏",
    layout="centered"
)

# Backend API URL from Streamlit Secrets
API_URL = st.secrets["API_URL"]

# Load dataset to get valid weight range
data = pd.read_csv("datasets/height_weight.csv")

min_weight = float(data["Weight"].min())
max_weight = float(data["Weight"].max())

# Application Title
st.title("📏 Weight to Height Prediction")

# Application Description
st.write(
    "This application uses a Machine Learning "
    "Linear Regression model to predict height based on weight."
)

st.divider()

# User Input Section
st.header("Enter Your Weight")

weight = st.number_input(
    "Enter your weight (kg)",
    min_value=min_weight,
    max_value=max_weight,
    value=60.0,
    step=1.0
)

# Prediction Button
if st.button("Predict Height"):

    # Prepare API request
    payload = {
        "weight": weight
    }

    try:
        # Send POST request to FastAPI
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )

        # Check API response
        response.raise_for_status()

        # Convert response into JSON
        result = response.json()

        # Extract predicted height
        predicted_height = result["predicted_height"]

        # Display Result
        st.success("Prediction Completed!")

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Height",
            value=f"{predicted_height:.2f} cm"
        )

        st.write(f"Your entered weight: **{weight:.1f} kg**")

    except requests.exceptions.RequestException as e:
        st.error(f"Backend API Error: {e}")

    except (KeyError, ValueError):
        st.error("Invalid response received from backend.")

st.divider()

# Model Information
st.header("About This Model")

st.write("""
- Algorithm: Linear Regression
- Feature: Weight
- Target: Height
- Dataset: Height and Weight Dataset
- Training Records: 24
- Testing Records: 6
""")

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)
