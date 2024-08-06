import streamlit as st
import joblib
import os
import numpy as np

# Ensure the model file is in the correct path
MODEL_FILE_PATH = r'C:\Users\A\Desktop\CAPSTONE\best_random_forest.pkl'

# Load the trained Random Forest model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE_PATH):
        # Add your model downloading code here if needed
        pass
    return joblib.load(MODEL_FILE_PATH)

model = load_model()

def predict_diabetes(inputs):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict_proba(input_array)
    return prediction[0]

# Streamlit app
st.title("Diabetes Prediction App")
st.write("This app predicts diabetes risk based on user input.")

# Define the questionnaire for input data
Sex = st.selectbox("Sex", ("Female", "Male"))
Age = st.selectbox("Age", [
    "18-24", "25-29", "30-34", "35-39", "40-44", 
    "45-49", "50-54", "55-59", "60-64", "65-69", 
    "70-74", "75-79", "Over 80 Years"
])
Education = st.selectbox("Education level", [
    "Never attended school or only kindergarten", "Elementary", 
    "Some High school", "High school graduate", 
    "Some college", "College graduate"
])
Income = st.selectbox("Income level", [
    "Less than $10000", "$10000-15000", "$15000-20000", "$20000-25000", 
    "$25000-35000", "$35000-50000", "$50000-75000", "Over $75000"
])
HighBP = st.selectbox("Do you have high blood pressure?", ("No", "Yes"))
HighChol = st.selectbox("Do you have high cholesterol?", ("No", "Yes"))
CholCheck = st.selectbox("Have you had your cholesterol checked?", ("No", "Yes"))
Weight = st.number_input("What is your weight in kilograms?", min_value=10.0, max_value=300.0, step=0.1)
Height = st.number_input("What is your height in meters?", min_value=0.5, max_value=2.5, step=0.01)
Smoker = st.selectbox("Are you a smoker?", ("No", "Yes"))
Stroke = st.selectbox("Have you had a stroke?", ("No", "Yes"))
HeartDiseaseorAttack = st.selectbox("Have you had heart disease or attack?", ("No", "Yes"))
PhysActivity = st.selectbox("Do you engage in physical activity?", ("No", "Yes"))
Fruits = st.selectbox("Do you consume fruits regularly?", ("No", "Yes"))
Veggies = st.selectbox("Do you consume vegetables regularly?", ("No", "Yes"))
HvyAlcoholConsump = st.selectbox("Do you consume heavy amounts of alcohol?", ("No", "Yes"))
AnyHealthcare = st.selectbox("Do you have any healthcare coverage?", ("No", "Yes"))
NoDocbcCost = st.selectbox("Have you been unable to see a doctor because of cost?", ("No", "Yes"))
DiffWalk = st.selectbox("Do you have difficulty walking?", ("No", "Yes"))
GenHlth = st.slider("General health (1=Excellent, 5=Poor)", 1, 5)
MentHlth = st.slider("Number of days with poor mental health in the past month", 0, 30)
PhysHlth = st.slider("Number of days with poor physical health in the past month", 0, 30)

# Calculate BMI
BMI = Weight / (Height ** 2)

# Mapping the descriptive text to binary values
Sex_binary = 0 if Sex == "Female" else 1
HighBP_binary = 0 if HighBP == "No" else 1
HighChol_binary = 0 if HighChol == "No" else 1
CholCheck_binary = 0 if CholCheck == "No" else 1
Smoker_binary = 0 if Smoker == "No" else 1
Stroke_binary = 0 if Stroke == "No" else 1
HeartDiseaseorAttack_binary = 0 if HeartDiseaseorAttack == "No" else 1
PhysActivity_binary = 0 if PhysActivity == "No" else 1
Fruits_binary = 0 if Fruits == "No" else 1
Veggies_binary = 0 if Veggies == "No" else 1
HvyAlcoholConsump_binary = 0 if HvyAlcoholConsump == "No" else 1
AnyHealthcare_binary = 0 if AnyHealthcare == "No" else 1
NoDocbcCost_binary = 0 if NoDocbcCost == "No" else 1
DiffWalk_binary = 0 if DiffWalk == "No" else 1

# Convert Age, Education, and Income to numerical values
Age_mapping = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, 
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, 
    "70-74": 11, "75-79": 12, "Over 80 Years": 13
}
Education_mapping = {
    "Never attended school or only kindergarten": 1, "Elementary": 2, 
    "Some High school": 3, "High school graduate": 4, 
    "Some college": 5, "College graduate": 6
}
Income_mapping = {
    "Less than $10000": 1, "$10000-15000": 2, "$15000-20000": 3, "$20000-25000": 4, 
    "$25000-35000": 5, "$35000-50000": 6, "$50000-75000": 7, "Over $75000": 8
}

Age_binary = Age_mapping[Age]
Education_binary = Education_mapping[Education]
Income_binary = Income_mapping[Income]

# Collect the input data
input_data = [
    Sex_binary, Age_binary, Education_binary, Income_binary, HighBP_binary, HighChol_binary, CholCheck_binary, 
    BMI, Smoker_binary, Stroke_binary, HeartDiseaseorAttack_binary, PhysActivity_binary, Fruits_binary, 
    Veggies_binary, HvyAlcoholConsump_binary, AnyHealthcare_binary, NoDocbcCost_binary, DiffWalk_binary, 
    GenHlth, MentHlth, PhysHlth
]

# Predict diabetes risk
if st.button("Submit Results"):
    probabilities = predict_diabetes(input_data)
    classes = ["No Diabetes", "Prediabetes", "Diabetes"]
    st.write("Prediction Probabilities:")
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {probabilities[i] * 100:.2f}%")
