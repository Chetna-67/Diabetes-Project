import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('diabetes.csv')
df = data.iloc[:500]
x = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
       'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']
model = RandomForestClassifier(n_estimators=300,
    random_state=42)
model.fit(x, y)

st.title("Diabetes Risk Predictor")

st.write("This app uses a Random Forest machine learning model to predict diabetes risk based on health data.")

if st.button("Check Model Accuracy"):
    last_df = data.iloc[500:]

    z = last_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

    value = model.predict(z)

    y_test = last_df['Outcome']

    accuracy = accuracy_score(y_test, value)

    st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=0, value=30)

if st.button("Predict"):
    new_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], columns=x.columns)

    result = model.predict(new_data)

    if result[0] == 1:
        st.error("Prediction: Higher diabetes risk")
    else:
        st.success("Prediction: Lower diabetes risk")
