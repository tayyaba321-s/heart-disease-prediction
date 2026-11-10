"""
Streamlit app for the Heart Disease Prediction project.

Run locally:  streamlit run app.py
Needs:        cardio_model.joblib (created by the notebook) in the same folder
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")


# Load the model once and keep it in memory so the app stays fast.
@st.cache_resource
def load_model():
    saved = joblib.load("cardio_model.joblib")
    return saved["model"], saved["features"]


model, features = load_model()

st.title(" Heart Disease Risk Prediction")
st.write("Enter the patient details and click **Predict**. "
         "This is a student project and is **not a medical diagnosis**.")

level_names = {1: "Normal", 2: "Above normal", 3: "Well above normal"}
yes_no = lambda x: "Yes" if x == 1 else "No"

# ---------------- Inputs that are always needed ----------------
# age, weight, ap_hi, ap_lo, cholesterol are model features;
# height is needed to calculate BMI.
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
    height = st.number_input("Height (cm)", min_value=120, max_value=210, value=165)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
with col2:
    ap_hi = st.number_input("Systolic BP (upper)", min_value=70, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (lower)", min_value=40, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol", options=[1, 2, 3],
                               format_func=lambda x: level_names[x])

# ---------------- Inputs shown only if the model uses them ----------------
# This keeps the app correct even if you retrain with different features:
# the user is never asked for something the model ignores.
gender, gluc, smoke, alco, active = 1, 1, 0, 0, 1   # placeholders, used only if in features
extra = [f for f in ["gender", "gluc", "smoke", "alco", "active"] if f in features]
if extra:
    st.markdown("**Additional details**")
    c1, c2 = st.columns(2)
    if "gender" in extra:
        gender = c1.selectbox("Gender", options=[1, 2],
                              format_func=lambda x: "Female" if x == 1 else "Male")
    if "gluc" in extra:
        gluc = c2.selectbox("Glucose", options=[1, 2, 3],
                            format_func=lambda x: level_names[x])
    if "smoke" in extra:
        smoke = c1.selectbox("Smoker?", options=[0, 1], format_func=yes_no)
    if "alco" in extra:
        alco = c2.selectbox("Drinks alcohol?", options=[0, 1], format_func=yes_no)
    if "active" in extra:
        active = c1.selectbox("Physically active?", options=[0, 1], format_func=yes_no)

# ---------------- Prediction ----------------
if st.button("Predict", type="primary"):
    if ap_hi <= ap_lo:
        st.error("Systolic BP must be higher than diastolic BP.")
    else:
        bmi = weight / (height / 100) ** 2
        row = pd.DataFrame([{
            "age": age, "gender": gender, "height": height, "weight": weight,
            "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
            "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
            "bmi": bmi, "pulse_pressure": ap_hi - ap_lo,
        }])[features]          # keep only the columns the model was trained on

        pred = model.predict(row)[0]
        prob = model.predict_proba(row)[0, 1]

        st.subheader("Result")
        if pred == 1:
            st.error(f"Higher risk of heart disease (probability: {prob:.1%})")
        else:
            st.success(f"Lower risk of heart disease (probability: {prob:.1%})")
        st.progress(float(prob))
        st.caption(f"BMI: {bmi:.1f}  |  Pulse pressure: {ap_hi - ap_lo}")

st.divider()
st.caption("Model accuracy is around 73% on the test set, and it misses roughly "
           "1 in 3 people who have the disease. Always consult a doctor for real "
           "medical decisions.")