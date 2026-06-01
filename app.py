import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Config
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    rnn = tf.keras.models.load_model("simple_rnn_model.keras")
    lstm = tf.keras.models.load_model("lstm_model.keras")
    gru = tf.keras.models.load_model("gru_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return rnn, lstm, gru, tokenizer

rnn_model, lstm_model, gru_model, tokenizer = load_models()

MAX_LEN = 200

# Header
st.markdown("<h1>🎬 Movie Review Sentiment Analysis System</h1>", unsafe_allow_html=True)

st.markdown(
    "<h3>Deep Learning Based Sentiment Classification</h3>",
    unsafe_allow_html=True
)

st.divider()

# Model Selection
selected_model = st.radio(
    "Choose Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# Review Input
review = st.text_area(
    "Enter your movie review here...",
    height=150
)

# Predict Function
def predict_review(text, model):
    seq = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    pred = float(model.predict(padded, verbose=0)[0][0])

    sentiment = "Positive" if pred >= 0.5 else "Negative"

    confidence = pred * 100 if pred >= 0.5 else (1 - pred) * 100

    return sentiment, confidence, pred

# Button
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        model_map = {
            "SimpleRNN": rnn_model,
            "LSTM": lstm_model,
            "GRU": gru_model
        }

        sentiment, confidence, prob = predict_review(
            review,
            model_map[selected_model]
        )

        st.success(f"Sentiment: {sentiment}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Probability Distribution")

        probs = pd.DataFrame({
            "Class": ["Negative", "Positive"],
            "Probability": [1-prob, prob]
        })

        st.bar_chart(
            probs.set_index("Class")
        )

        st.divider()

        st.subheader("Model Comparison")

        comparison = []

        for name, model in model_map.items():

            sent, conf, p = predict_review(
                review,
                model
            )

            comparison.append([
                name,
                sent,
                round(conf, 2)
            ])

        comp_df = pd.DataFrame(
            comparison,
            columns=[
                "Model",
                "Sentiment",
                "Confidence (%)"
            ]
        )

        st.dataframe(
            comp_df,
            use_container_width=True
        )