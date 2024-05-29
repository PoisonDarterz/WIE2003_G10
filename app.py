import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('rf_model.pkl')

# Create a function to get user input
def get_user_input():
    # Create user input fields
    age = st.sidebar.slider('Age', 20, 100, 50)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3, 4])
    resting_bp = st.sidebar.slider('Resting Blood Pressure', 80, 200)
    cholesterol = st.sidebar.slider('Cholesterol', 100, 600)
    fasting_blood_sugar = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    resting_ecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    max_heart_rate = st.sidebar.slider('Max Heart Rate Achieved', 60, 220)
    exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('Oldpeak', 0.0, 4.0, step=0.1)
    st_slope = st.sidebar.selectbox('ST Slope', [1, 2, 3])

    # Store a dictionary into a data frame
    user_data = {
        'Age': age,
        'Sex': sex,
        'Chest Pain Type': chest_pain_type,
        'Resting Blood Pressure': resting_bp,
        'Cholesterol': cholesterol,
        'Fasting Blood Sugar > 120 mg/dl': fasting_blood_sugar,
        'Resting Electrocardiographic Results': resting_ecg,
        'Max Heart Rate Achieved': max_heart_rate,
        'Exercise Induced Angina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST Slope': st_slope
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Scale the user input
scaler = StandardScaler()
user_input_scaled = scaler.transform(user_input)

# Make predictions
prediction = model.predict(user_input_scaled)

# Display the predictions
st.subheader('Prediction:')
heart_disease = ['No', 'Yes']
st.write(heart_disease[prediction[0]])