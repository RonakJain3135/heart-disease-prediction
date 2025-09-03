import streamlit as st
import joblib
import pandas as pd

# Load model and training columns
model = joblib.load("HeartDiseasePredictorV1.joblib")
columns = joblib.load("heart_columns.joblib")

st.title("Heart Disease Risk Predictor")

st.markdown("""
This tool predicts the likelihood of **heart disease** based on patient health data.  
Please fill in the details below:
""")

# Collect user inputs with explanations
age = st.number_input("Age", 18, 100, 50, help="Age of the person in years")
sex = st.selectbox("Sex", ["M", "F"], help="Biological sex: M = Male, F = Female")
chest = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"],
    help="Types of chest pain:\n- ATA: Typical Angina\n- NAP: Non-Anginal Pain\n- ASY: Asymptomatic\n- TA: Typical Angina"
)
restbp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Resting blood pressure in mm Hg")
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, help="Serum cholesterol level in mg/dl")
fasting = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], help="1 = True, 0 = False")
restecg = st.selectbox(
    "Resting ECG Results",
    ["Normal", "ST", "LVH"],
    help="Electrocardiogram results:\n- Normal\n- ST: ST-T wave abnormality\n- LVH: Left Ventricular Hypertrophy"
)
maxhr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150, help="Maximum heart rate achieved during exercise")
angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], help="Y = Yes, N = No")
oldpeak = st.number_input("Oldpeak (ST depression)", -2.0, 7.0, 1.0, step=0.1, help="ST depression induced by exercise relative to rest")
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="The slope of the peak exercise ST segment")

if st.button("Predict"):
    # Prepare input
    user_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest,
        "RestingBP": restbp,
        "Cholesterol": chol,
        "FastingBS": fasting,
        "RestingECG": restecg,
        "MaxHR": maxhr,
        "ExerciseAngina": angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }
    input_df = pd.DataFrame([user_data])
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.error("Prediction: High risk of Heart Disease")
    else:
        st.success("Prediction: Low risk of Heart Disease")
print(model.predict_proba(input_encoded))
