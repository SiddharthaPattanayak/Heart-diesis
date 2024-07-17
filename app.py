import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"C:\Users\LENOVO\Desktop\New folder\HeartDeasis_Project\heart.csv")

# Split the data into features and target variable
X = df.drop(columns='target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
sv = SVC()
sv.fit(X_train, y_train)

# Calculate accuracy
X_train_Predict = sv.predict(X_train)
X_train_acc = accuracy_score(X_train_Predict, y_train)
X_test_Predict = sv.predict(X_test)
X_test_acc = accuracy_score(X_test_Predict, y_test)

# Streamlit App
st.title("Heart Disease Prediction")

st.write(f"Training Accuracy: {X_train_acc:.2f}")
st.write(f"Testing Accuracy: {X_test_acc:.2f}")

st.header("Enter your details to predict heart disease")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholestoral in mg/dl (chol)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by flourosopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (thal)", [0, 1, 2, 3])

# Prepare input data
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
input_data_as_numpy = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy.reshape(1, -1)
std_data = sc.transform(input_data_reshaped)

# Predict
prediction = sv.predict(std_data)

# Display the result
if st.button("Predict"):
    if prediction[0] == 1:
        st.write("The person is likely to have heart disease.")
    else:
        st.write("The person is unlikely to have heart disease.")
