# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:43:04 2024

@author: A
"""

import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
@st.cache_resource
def load_model():
    return joblib.load('best_random_forest.pkl')

model = load_model()

def predict_diabetes(inputs):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit app
st.title("Diabetes Prediction App")
st.write("This app predicts diabetes risk based on user input.")

# Define the questionnaire for input data
HighBP = st.selectbox("Do you have high blood pressure?", (0, 1))
HighChol = st.selectbox("Do you have high cholesterol?", (0, 1))
CholCheck = st.selectbox("Have you had your cholesterol checked?", (0, 1))
BMI = st.slider("What is your BMI?", 10, 50)
Smoker = st.selectbox("Are you a smoker?", (0, 1))
Stroke = st.selectbox("Have you had a stroke?", (0, 1))
HeartDiseaseorAttack = st.selectbox("Have you had heart disease or attack?", (0, 1))
PhysActivity = st.selectbox("Do you engage in physical activity?", (0, 1))
Fruits = st.selectbox("Do you consume fruits regularly?", (0, 1))
Veggies = st.selectbox("Do you consume vegetables regularly?", (0, 1))
HvyAlcoholConsump = st.selectbox("Do you consume heavy amounts of alcohol?", (0, 1))
AnyHealthcare = st.selectbox("Do you have any healthcare coverage?", (0, 1))
NoDocbcCost = st.selectbox("Have you been unable to see a doctor because of cost?", (0, 1))
GenHlth = st.slider("General health (1=Excellent, 5=Poor)", 1, 5)
MentHlth = st.slider("Number of days with poor mental health in the past month", 0, 30)
PhysHlth = st.slider("Number of days with poor physical health in the past month", 0, 30)
DiffWalk = st.selectbox("Do you have difficulty walking?", (0, 1))
Sex = st.selectbox("Sex (0=Female, 1=Male)", (0, 1))
Age = st.slider("Age", 18, 80)
Education = st.slider("Education level (1-6)", 1, 6)
Income = st.slider("Income level (1-8)", 1, 8)

# Collect the input data
input_data = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity,
              Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth,
              PhysHlth, DiffWalk, Sex, Age, Education, Income]

# Predict diabetes risk
if st.button("Predict"):
    prediction = predict_diabetes(input_data)
    if prediction == 1:
        st.write("Prediction: Diabetic")
    elif prediction == 0:
        st.write("Prediction: Non-Diabetic")
    else:
        st.write("Prediction: Pre-Diabetic")
