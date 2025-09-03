**Heart Disease Risk Predictor**

A machine learning-powered web app built with Streamlit that predicts the likelihood of heart disease based on clinical and lifestyle parameters.

This project uses a Decision Tree / Random Forest classifier trained on the Heart Failure Prediction dataset
. Users can enter their health data (e.g., age, cholesterol, blood pressure) and get a prediction of whether they are at low or high risk of heart disease.


**🚀 Features**

Interactive Streamlit web app with user-friendly input fields.

Predicts heart disease risk (low/high) in real time.

Displays confidence scores for predictions.

Easy to deploy on Streamlit Community Cloud.

Model trained with scikit-learn using cleaned and preprocessed Kaggle dataset.


**🧑‍⚕️ Input Parameters**

The app considers the following features:

Age: Age of the person in years

Sex: Male (M) / Female (F)

Chest Pain Type: ATA, NAP, ASY, TA

Resting Blood Pressure: mm Hg at rest

Cholesterol: Serum cholesterol (mg/dl)

Fasting Blood Sugar: 1 = >120 mg/dl, 0 = otherwise

Resting ECG: Normal, ST, LVH

Max HR: Maximum heart rate achieved

Exercise-Induced Angina: Y / N

Oldpeak: ST depression (exercise vs rest)

ST Slope: Up, Flat, Down


**🛠️ Tech Stack**

Python

scikit-learn

pandas

numpy

joblib

Streamlit
