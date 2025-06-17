import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentiment_analysis.data_preprocessing import preprocess_text # Ensure this is consistent with training

# Configuration for prediction
MODEL_PATH = 'sentiment_analysis/sentiment_lstm_model.h5'
TOKENIZER_PATH = 'sentiment_analysis/tokenizer.pkl'
MAX_SEQUENCE_LENGTH = 256 # Must match max_len used during training

def load_model_and_tokenizer():
    """Loads the trained Keras model and tokenizer."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure 'model_training.py' has been run to save the model and tokenizer.")
        return None, None

def predict_sentiment(text, model, tokenizer, max_len=MAX_SEQUENCE_LENGTH):
    """
    Predicts the sentiment of a given text.
    """
    if model is None or tokenizer is None:
        print("Model or tokenizer not loaded. Cannot predict.")
        return "Error: Model not loaded"

    # Preprocess the input text consistently
    processed_text = preprocess_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

    # Predict
    prediction = model.predict(padded_sequence)[0][0]

    # Interpret prediction
    if prediction >= 0.5:
        return "Positive", prediction
    else:
        return "Negative", prediction

if __name__ == "__main__":
    # Load model and tokenizer once
    model, tokenizer = load_model_and_tokenizer()

    if model and tokenizer:
        print("\n--- Sentiment Prediction Script ---")
        while True:
            user_input = input("Enter a movie review (type 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            sentiment, probability = predict_sentiment(user_input, model, tokenizer)
            print(f"Review: \"{user_input}\"")
            print(f"Predicted Sentiment: {sentiment} (Probability: {probability:.4f})\n")
    else:
        print("Exiting due to model/tokenizer loading error.")