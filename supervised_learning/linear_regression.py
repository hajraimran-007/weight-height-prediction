
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv("datasets/height_weight.csv")

# Display first 5 records
print(data.head())

# Check dataset dimensions
print(data.shape)

# Check column names and data types
print(data.info())

# Statistical summary
print(data.describe())

# Check missing values
print("Missing Values:")
print(data.isnull().sum())

# Check duplicate records
print("Duplicate Records:")
print(data.duplicated().sum())


# Data Preprocessing

# Remove duplicate records
data = data.drop_duplicates()

print("Updated Shape:", data.shape)


# Separate Features and Target

X = data[["Weight"]]
y = data["Height"]

print("Features (X):")
print(X)

print("Target (y):")
print(y)

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# Split Dataset into Training and Testing Sets

# test_size=0.2: 20% testing data
# random_state=42: Reproducible random split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)


# Create Linear Regression Model

model = LinearRegression()


# Train the Model

model.fit(X_train, y_train)

print("Model Training Completed!")


# Check Slope and Intercept

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


# Save the Trained Model

joblib.dump(model, "weight_height_model.joblib")

print("Model Saved Successfully!")


# Make Predictions

y_pred = model.predict(X_test)

print("Actual Values:")
print(y_test.values)

print("Predicted Values:")
print(y_pred)


# Calculate Prediction Errors

errors = y_pred - y_test.values

print("Prediction Errors:")
print(errors)


# User Input Prediction

user_weight = float(input("Enter your weight: "))

min_weight = data["Weight"].min()
max_weight = data["Weight"].max()

if min_weight <= user_weight <= max_weight:

    user_input = pd.DataFrame({
        "Weight": [user_weight]
    })

    predicted_height = model.predict(user_input)

    print("Your Weight:", user_weight)
    print("Predicted Height:", predicted_height[0])

else:
    print(
        "Please enter a weight between",
        min_weight,
        "and",
        max_weight
    )
