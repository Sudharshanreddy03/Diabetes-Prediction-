import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading dataset
data = pd.read_csv('health care diabetes.csv')

# Replacing null values with the mean
data['Glucose'] = data['Glucose'].replace([0], [data['Glucose'].mean()])
data['BloodPressure'] = data['BloodPressure'].replace([0], [data['BloodPressure'].mean()])
data['SkinThickness'] = data['SkinThickness'].replace([0], [data['SkinThickness'].mean()])
data['Insulin'] = data['Insulin'].replace([0], [data['Insulin'].mean()])

# Features and target variable
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# Split the data into training and testing sets
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the model
logreg = LogisticRegression(solver='liblinear', random_state=123)
logreg.fit(trainx, trainy)

# Streamlit UI
st.title("Diabetes Prediction Model")

st.sidebar.header("User Input Features")

# Collecting user input features
def user_input_features():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 199, 117)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
    insulin = st.sidebar.slider("Insulin", 0, 846, 30)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
    age = st.sidebar.slider("Age", 21, 81, 29)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }

    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

st.subheader("User Input:")
st.write(user_input)

# Make predictions
prediction = logreg.predict(user_input)

st.subheader("Prediction:")
if prediction[0] == 1:
    st.write("The model predicts that the person has diabetes.")
else:
    st.write("The model predicts that the person does not have diabetes.")

