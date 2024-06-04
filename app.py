import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')
# Load your dataset
dataset = pd.read_csv('cleaned_dataset.csv')

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
prediction_proba = model.predict_proba(user_input_scaled)

# GUI
# Create tabs
tab1, tab2 = st.tabs(['Predictions', 'Dataset'])

with tab1:
    # Display the predictions
    st.subheader('Prediction:')
    heart_disease = ['No', 'Yes']
    st.write(heart_disease[prediction[0]])

    # Display prediction probability
    st.subheader('Prediction Probability:')
    st.write(f'No: {prediction_proba[0][0]*100:.2f}%')
    st.write(f'Yes: {prediction_proba[0][1]*100:.2f}%')

    # Display a bar chart for prediction probabilities
    st.subheader('Prediction Probability Chart:')
    st.bar_chart(prediction_proba[0])

    # Explanation of results
    if prediction[0] == 1:
        st.write("The model predicts that there is a high chance of heart disease.")
    else:
        st.write("The model predicts that there is a low chance of heart disease. Keep maintaining a healthy lifestyle!")

with tab2:
    # Display the dataset
    st.subheader('Dataset')
    st.dataframe(dataset)