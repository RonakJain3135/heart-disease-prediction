import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load and preprocess data
df = pd.read_csv("heart.csv")
df.drop_duplicates(inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns="HeartDisease")
y = df_encoded["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train Decision Tree
model = DecisionTreeClassifier(random_state=42, max_depth=4) 
model.fit(X_train, y_train)

# Save columns so we can align user input
columns = X.columns

# 3. Streamlit App
st.title("Heart Disease Risk Predictor")

st.markdown("""
This tool predicts the likelihood of **heart disease** based on health and lifestyle factors.  
Please fill in the details below.  
*(Note: This app is for learning purposes only and not medical advice.)*
""")

# Collect user inputs
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
chest = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
st.markdown("""
**Chest Pain Type**:  
- **ATA**: Atypical Angina (mild chest pain, less linked to heart disease)  
- **NAP**: Non-Anginal Pain (chest pain unrelated to the heart)  
- **ASY**: Asymptomatic (no pain, but often high risk in data)  
- **TA**: Typical Angina (classic chest pain linked with heart disease)  
""")
restbp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
st.markdown("""
**Resting ECG (Electrocardiogram)**:  
- **Normal**: Normal heart activity  
- **ST**: Abnormal ST-T wave patterns (possible issues)  
- **LVH**: Signs of Left Ventricular Hypertrophy (enlarged heart muscle)  
""")
maxhr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
st.markdown("**Exercise-Induced Angina**: Y = chest pain during exercise, N = no pain.")
oldpeak = st.number_input("Oldpeak (ST depression)", -2.0, 7.0, 1.0, step=0.1)
st.markdown("**Oldpeak**: Drop in ST segment on ECG after exercise. Higher values = more risk.")
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
st.markdown("""
**ST Slope** (shape of ECG line after exercise):  
- **Up**: Normal, healthy slope  
- **Flat**: Intermediate concern  
- **Down**: Often linked with higher heart disease risk  
""")

if st.button("Predict"):
    # Prepare user input
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

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][prediction]

    if prediction == 1:
        st.error(f"Prediction: High risk of Heart Disease (Confidence: {proba:.2f})")
    else:
        st.success(f"Prediction: Low risk of Heart Disease (Confidence: {proba:.2f})")
