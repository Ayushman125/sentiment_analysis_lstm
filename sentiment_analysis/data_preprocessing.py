import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd # Only if loading from CSV
import tensorflow as tf
# Download NLTK resources if not already downloaded
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

def preprocess_text(text):
    """
    Applies a series of text preprocessing steps.
    """
    text = text.lower() # Lowercasing
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english')]) # Remove stopwords and tokenize
    return text

def load_and_preprocess_imdb_dataset(num_words=10000, max_len=256):
    """
    Loads the IMDb dataset, preprocesses text, and prepares sequences.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

    word_index = tf.keras.datasets.imdb.get_word_index()
    index_to_word = {value: key for key, value in word_index.items()}

    # Convert integer sequences back to text for preprocessing
    train_texts = [" ".join([index_to_word.get(i - 3, "?") for i in review]) for review in x_train]
    test_texts = [" ".join([index_to_word.get(i - 3, "?") for i in review]) for review in x_test]

    print("Preprocessing training texts...")
    processed_train_texts = [preprocess_text(text) for text in train_texts]
    print("Preprocessing testing texts...")
    processed_test_texts = [preprocess_text(text) for text in test_texts]

    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(processed_train_texts)

    train_sequences = tokenizer.texts_to_sequences(processed_train_texts)
    test_sequences = tokenizer.texts_to_sequences(processed_test_texts)

    train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

    return train_padded, y_train, test_padded, y_test, tokenizer, max_len

# Example for custom CSV dataset (if not using IMDb built-in)
def load_and_preprocess_custom_dataset(filepath, text_column, label_column, num_words=10000, max_len=256):
    df = pd.read_csv(filepath)
    df[text_column] = df[text_column].apply(preprocess_text)

    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(df[text_column])

    sequences = tokenizer.texts_to_sequences(df[text_column])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    labels = df[label_column].values

    # Split data (you might use train_test_split from sklearn)
    # For simplicity, returning all here, you'd split before training
    return padded_sequences, labels, tokenizer, max_len