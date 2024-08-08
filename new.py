import streamlit as st
import joblib
import os
import numpy as np
from PIL import Image

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

# URLs of the top image and background image
top_image_url = 'https://images.pexels.com/photos/6823764/pexels-photo-6823764.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
background_image_url = 'https://images.unsplash.com/photo-1663601398716-3d40cef5d1fc?q=80&w=2127&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'

# Set up the landing page
def landing_page():
    st.markdown(
        f"""
        <style>
        .main {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .reportview-container .main .block-container {{
            padding-top: 2rem;
            padding-right: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
        }}
        body, .reportview-container, .markdown-text-container {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.image(top_image_url, use_column_width=True)
    
    st.title("Welcome to the Diabetes Prediction App")
    
    st.write(
        """
        **A little about Diabetes:** **🍎**

        Diabetes is a chronic condition that affects how the body processes food for energy. There are different types of diabetes:

        Type 1 Diabetes: This type occurs when the body's immune system attacks and destroys the cells in the pancreas that produce insulin, a hormone needed to regulate blood sugar. People with Type 1 diabetes need to take insulin every day.

        Type 2 Diabetes: This is the most common form of diabetes and happens when the body either becomes resistant to insulin or doesn’t produce enough insulin. It's often linked to lifestyle factors and can sometimes be managed with diet, exercise, and medication.
        
        Prediabetes: Prediabetes is a condition where blood sugar levels are higher than normal but not yet high enough to be diagnosed as Type 2 diabetes. It serves as a warning sign and a critical window for intervention. Lifestyle changes such as increased physical activity, a healthy diet, and weight loss can help manage prediabetes and reduce the risk of progressing to Type 2 diabetes.

        Gestational Diabetes: This type occurs during pregnancy and affects how cells use sugar, leading to high blood sugar levels. It usually disappears after childbirth but increases the risk of developing Type 2 diabetes later in life.        
       
        **Diabetes in Kenya: A Snapshot** **📸**

        Diabetes is a major global health problem. In 2021, it caused 6.7 million deaths and cost over $960 billion worldwide. This issue is growing fast, with diabetes cases expected to increase by 46% by 2045. Type 2 diabetes, which makes up 90-95% of diabetes cases, can often be prevented by maintaining a healthy weight, eating a balanced diet, and staying active.

        Sub-Saharan Africa, including Kenya, is seeing a dramatic rise in diabetes. The number of diabetes cases in this region is predicted to increase 2.5 times by 2045, with health costs rising from usd 12.6 billion to usd 46.7 billion. In Kenya, diabetes has been a major concern since 2015. Surveys show that type 2 diabetes is more common in urban areas (3.4%) compared to rural areas (1.9%), and even higher in low-income neighborhoods in Nairobi (4.1-5.3%). Diabetes-related deaths in Nairobi have increased by 65% from 2009 to 2019, making diabetes one of the top 10 causes of death and disability in Kenya.

        The high diabetes rates in Nairobi might be due to delayed diagnoses and risk factors like obesity, poor diets, and lack of exercise. In Nairobi, 38% of people are overweight, compared to the national average of 28%, and many residents do not eat enough fruits and vegetables. While many Kenyans report being physically active, urban residents tend to be less active than those in rural areas. To tackle these issues, we need local, community-based prevention efforts to effectively reduce the growing diabetes problem.
        
        **Purpose of This Questionnaire:** **🔍**
        This questionnaire is designed to collect relevant health information to assess your risk of developing diabetes. The data collected will help us make accurate predictions based on our trained model.
        
        **How We Use Your Information:** **👩🏽‍🔬**
        We ask for various details such as age, weight, height, and lifestyle habits to assess your health status. Our model uses this data to predict whether you are diabetic, pre-diabetic, or non-diabetic. This is also accompanied by your BMI results a health score analysis.
        """
    )
    
    # Button to go to the deployment page
    if st.button("Proceed to Questionnaire 👆🏽"):
        st.session_state.page = "questionnaire"
    
    # FAQ Section
    st.write(
        """
        **FAQ: How Does the Questionnaire Work?**
        
        - **1. What information do you need to provide for the questionnaire?**
        
        You need to provide details such as your sex, age, education level, income, and information about your health and lifestyle, including whether you have high blood pressure, high cholesterol, or any other health conditions.
        - **2. How is my information used to predict diabetes risk?** 
        
        The information you provide is processed and used as input to a trained machine learning model that assesses your risk of diabetes based on patterns learned from similar data.
        - **3. What should I do if I’m unsure about how to answer a question?**
        
        If you’re unsure about how to answer a question, try to select the option that best describes your situation. If you’re still uncertain, you can consult a healthcare professional for advice.
        - **4. How accurate are the predictions made by the model?**
        
        The model provides probabilities based on the data you provide. While it can offer insights at 93% accuracy, it is important to follow up with a healthcare professional for a comprehensive assessment of your diabetes status.
        - **5. What happens if I don’t complete the questionnaire?**
        
        If you don’t complete the questionnaire, you won’t receive an accurate prediction. Make sure to answer all the questions to get the most accurate results.
        
        **FAQ: How Does the Machine Learning Model Work?**

        - **Data Collection:** We collect information through a series of questions about your health and lifestyle.
        - **Data Processing:** The information is processed and converted into a format that our model can understand.
        - **Model Prediction:** Our trained model analyzes the data and provides probabilities for different diabetes risk levels.
        - **Result:** Based on the analysis, the app predicts whether you are diabetic, pre-diabetic, or non-diabetic.
        """
    )
    
    


# Set up the prediction page
def prediction_page():
    st.markdown(
        f"""
        <style>
        .main {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .reportview-container .main .block-container {{
            padding-top: 2rem;
            padding-right: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
        }}
        body, .reportview-container, .markdown-text-container {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.image(top_image_url, use_column_width=True)
    
    st.title("Diabetes Prediction App")
    st.write("This app predicts and classifies diabetes risk based on user input.")
    
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
    
    # Calculate HealthScore
    HealthScore = np.mean([GenHlth, MentHlth, PhysHlth])
    
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
    
    # Print the length of the input data to debug
    st.write(f"Number of features in input data: {len(input_data)}")
    
    # Predict diabetes risk
    if st.button("Submit Results"):
        probabilities = predict_diabetes(input_data)
        classes = ["Diabetic", "Pre-Diabetic", "Non-Diabetic"] 
        
        # Display the prediction probabilities
        st.markdown("<h3 style='color:white;'>Prediction Probabilities:</h3>", unsafe_allow_html=True)
        for i, class_name in enumerate(classes):
            st.markdown(f"<p style='color:white;'>{class_name}: {probabilities[i] * 100:.2f}%</p>", unsafe_allow_html=True)
        
        # Display BMI and Health Score
        st.markdown(f"<p style='color:white;'>Your calculated BMI is: {BMI:.2f}</p>", unsafe_allow_html=True)
        
        # Health Score interpretation
        if HealthScore < 2:
            health_status = "Excellent health"
        elif HealthScore < 3:
            health_status = "Good health"
        elif HealthScore < 4:
            health_status = "Average health"
        elif HealthScore < 5:
            health_status = "Poor health"
        else:
            health_status = "Very poor health"
    
        st.markdown(f"<p style='color:white;'>Your Health Score is: {HealthScore:.2f} ({health_status})</p>", unsafe_allow_html=True)
        
        # Descriptor for HealthScore
        st.markdown("""  
                 <h4 style='color:white;'>HealthScore Interpretation</h4>
                 <b>HealthScore is calculated using an average of your general health score, mental health scores and physical health scores </b>
        <ul style='color:white;'>
        <li><b>1.0 - 2.0:</b> Excellent health</li>
        <li><b>2.0 - 3.0:</b> Good health</li>
        <li><b>3.0 - 4.0:</b> Average health</li>
        <li><b>4.0 - 5.0:</b> Poor health</li>
        <li><b>5.0+:</b> Very poor health</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Back button to return to the landing page
    if st.button("Back to Main Page"):
        st.session_state.page = "landing"

# Streamlit app
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

if st.session_state.page == 'landing':
    landing_page()
elif st.session_state.page == 'questionnaire':
    prediction_page()
