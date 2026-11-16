
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/weight_height_model.joblib")

# Load dataset to get valid weight range
data = pd.read_csv("datasets/height_weight.csv")

min_weight = float(data["Weight"].min())
max_weight = float(data["Weight"].max())

# Page Configuration
st.set_page_config(
    page_title="Weight to Height Prediction",
    page_icon="📏",
    layout="centered"
)

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

    # Convert input into DataFrame
    user_input = pd.DataFrame({
        "Weight": [weight]
    })

    # Make Prediction
    predicted_height = model.predict(user_input)

    # Display Result
    st.success("Prediction Completed!")

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Height",
        value=f"{predicted_height[0]:.2f} cm"
    )

    st.write(f"Your entered weight: **{weight:.1f} kg**")

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

st.caption("Developed using Python, Pandas, Scikit-learn and Streamlit.")
