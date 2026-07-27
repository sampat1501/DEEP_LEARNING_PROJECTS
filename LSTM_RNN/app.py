import pickle
from tensorflow.keras.models import load_model
import numpy as np

import streamlit as st

##Load lstm model
model = load_model("lstm_hamlet_rnn.h5")
##Load the tokeniser
with open("tokeniser.pickle", "rb") as file:
    tokeniser = pickle.load(file)

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


def predict_next_word(model, tokenizer, text, max_sequence_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1) :]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None


##Write the streamlit

# streamlit app

st.title("Next Word Prediction With LSTM And Early Stopping")

input_text = st.text_input("Enter the sequence of Words", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve max sequence length
    next_word = predict_next_word(model, tokeniser, input_text, max_sequence_len)

    st.write(f"Next word: {next_word}")
