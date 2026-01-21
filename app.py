import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

model = joblib.load("logistic_titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
sex_male = st.selectbox("Gender", ["Female", "Male"]) == "Male"

input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    st.success("✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive")
