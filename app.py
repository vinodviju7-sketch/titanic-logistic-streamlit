import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

# Load trained model
model = joblib.load("logistic_titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction")

# -----------------------------
# User Inputs
# -----------------------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents / Children Aboard", 0, 10, 0)

sex = st.selectbox("Gender", ["Female", "Male"])
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# -----------------------------
# Manual Encoding (same logic as training)
# -----------------------------
input_dict = {
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Sex_male": 1 if sex == "Male" else 0,
    "Embarked_Q": 1 if embarked == "Q" else 0,
    "Embarked_S": 1 if embarked == "S" else 0,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# -----------------------------
# ⭐ CRITICAL FIX ⭐
# Align input features EXACTLY with model training features
# -----------------------------
expected_features = model.feature_names_in_

input_df = input_df.reindex(columns=expected_features, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Did Not Survive (Probability: {probability:.2f})")
