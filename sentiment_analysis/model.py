import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

def build_lstm_model(vocab_size, embedding_dim=100, max_len=256):
    """
    Builds and compiles an LSTM model for sentiment classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(units=128, return_sequences=True)), # Using Bidirectional LSTM for better context
        Dropout(0.3),
        LSTM(units=64),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Trains the provided LSTM model.
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy