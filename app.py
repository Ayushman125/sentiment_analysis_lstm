import streamlit as st
from sentiment_analysis.predict import load_model_and_tokenizer, predict_sentiment
import os

# Load the model and tokenizer only once when the app starts
@st.cache_resource
def get_model_and_tokenizer():
    return load_model_and_tokenizer()

model, tokenizer = get_model_and_tokenizer()

st.set_page_config(page_title="LSTM Sentiment Analyzer", page_icon="🎬")

st.title("🎬 LSTM Movie Review Sentiment Analyzer")
st.markdown("""
Welcome to the LSTM-powered sentiment analysis tool!
Enter a movie review below to find out if it's positive or negative.
""")

if model is None or tokenizer is None:
    st.error("Error: Model or tokenizer could not be loaded. Please ensure 'model_training.py' has been run.")
else:
    user_input = st.text_area("Enter your movie review here:", height=150)

    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing..."):
                sentiment, probability = predict_sentiment(user_input, model, tokenizer)
                st.write("---")
                if sentiment == "Positive":
                    st.success(f"**Predicted Sentiment: Positive 👍**")
                else:
                    st.error(f"**Predicted Sentiment: Negative 👎**")
                st.info(f"Confidence: {probability:.2%}")
        else:
            st.warning("Please enter some text to analyze.")

st.markdown("""
---
*Built with TensorFlow/Keras and Streamlit by Your Name.*
""")