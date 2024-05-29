import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('rf_model.pkl')

def get_user_input():
    # Create user input fields
    age = st.sidebar.slider('age', 20, 100, 50)
    sex = st.sidebar.selectbox('sex', ['Male', 'Female'])
    chest_pain_type = st.sidebar.selectbox('chest pain type', [0, 1, 2, 3, 4])
    resting_bp = st.sidebar.slider('resting bp s', 80, 200)
    cholesterol = st.sidebar.slider('cholesterol', 100, 600)
    fasting_blood_sugar = st.sidebar.selectbox('fasting blood sugar', [0, 1])
    resting_ecg = st.sidebar.selectbox('resting ecg', [0, 1, 2])
    max_heart_rate = st.sidebar.slider('max heart rate', 60, 220)
    exercise_angina = st.sidebar.selectbox('exercise angina', [0, 1])
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 4.0, step=0.1)
    st_slope = st.sidebar.selectbox('ST slope', [1, 2, 3])

    sex = 1 if sex == 'Male' else 0

    # Store a dictionary into a data frame
    user_data = {
        'age': age,
        'sex': sex,
        'chest pain type': chest_pain_type,
        'resting bp s': resting_bp,
        'cholesterol': cholesterol,
        'fasting blood sugar': fasting_blood_sugar,
        'resting ecg': resting_ecg,
        'max heart rate': max_heart_rate,
        'exercise angina': exercise_angina,
        'oldpeak': oldpeak,
        'ST slope': st_slope
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Scale the user input
scaler = joblib.load('scaler.pkl')
user_input_scaled = scaler.transform(user_input)

# Make predictions
prediction = model.predict(user_input_scaled)

# Display the predictions
st.subheader('Prediction:')
heart_disease = ['No', 'Yes']
st.write(heart_disease[prediction[0]])