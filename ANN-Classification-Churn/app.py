import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

## Step 1: Load trained model

model = tf.keras.models.load_model("model.h5")

###Step 2: Load all the pickle files

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


##Step 3: Start strealit app
st.title("Customer Churn Prediction")
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 90, 1)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("NUmber of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

##Step 4: Prepare the input data
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Balance": [balance],
        "Tenure": [tenure],
        "NumOfProducts": [num_of_products],
        "EstimatedSalary": [estimated_salary],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
    }
)

## Step 5: one hot encoded geo value
encoded_geo = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    encoded_geo, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

##Step 6: Combine one hot encoded data with input data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Reorder columns exactly as used during scaler training

input_data = input_data[scaler.feature_names_in_]

##Step 7 : Scale the data

input_scaled = scaler.transform(input_data)

##Prediction
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]
st.write("The Prediction probability is : ", prediction_prob)


if prediction_prob > 0.5:
    st.write("Cutomer likely to churn")
else:
    st.write("The Customer is not likely to churn")
