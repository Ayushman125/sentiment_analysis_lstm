import tensorflow as tf
from sentiment_analysis.data_preprocessing import load_and_preprocess_imdb_dataset, preprocess_text
from sentiment_analysis.model import build_lstm_model, train_model, evaluate_model
import os
import pickle
import nltk

# --- GPU Check ---
print("--- Checking for GPU availability ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU (if multiple are present)
        # Or remove this block if you want TensorFlow to use all available GPUs
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU(s)")
        print("TensorFlow will use GPU!")
        # You can also print details about the GPU
        # for gpu in gpus:
        #     print(f"Name: {gpu.name}, Type: {gpu.device_type}")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU detected. TensorFlow will use CPU.")
print("-----------------------------------")
# --- End GPU Check ---

# Configuration
VOCAB_SIZE = 10000
MAX_LEN = 256
EMBEDDING_DIM = 100
EPOCHS = 10
BATCH_SIZE = 64
MODEL_SAVE_PATH = 'sentiment_analysis/sentiment_lstm_model.h5'
TOKENIZER_SAVE_PATH = 'sentiment_analysis/tokenizer.pkl'

def main():
    print("Loading and preprocessing IMDb dataset...")
    X_train, y_train, X_test, y_test, tokenizer, max_len = load_and_preprocess_imdb_dataset(
        num_words=VOCAB_SIZE, max_len=MAX_LEN
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("Building LSTM model...")
    model = build_lstm_model(VOCAB_SIZE, EMBEDDING_DIM, max_len)
    model.summary()

    print("Training model...")
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    print(f"Saving tokenizer to {TOKENIZER_SAVE_PATH}...")
    with open(TOKENIZER_SAVE_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Training and saving complete.")

if __name__ == "__main__":
    main()