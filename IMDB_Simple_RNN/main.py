# Import all Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

###Find the word index available in the data

word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load model
from tensorflow.keras.models import load_model

model = load_model("simple_rnn_imdb.h5")
model.summary()


# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [word_index.get(word, 2) + 3 for word in words]

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)

    return padded_review


### Prediction Function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]


##Design streamlit app

import streamlit as st

# Streamlit App
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

# User Input
user_input = st.text_area("Movie Review")

if st.button("Classify"):

    sentiment, score = predict_sentiment(user_input)

    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Prediction Score:** {score:.4f}")

    if sentiment == "Positive":
        st.success("✅ Positive Review")
    else:
        st.error("❌ Negative Review")
