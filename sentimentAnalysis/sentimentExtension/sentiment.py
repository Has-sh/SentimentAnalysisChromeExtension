from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.layers import SpatialDropout1D, LSTM
from sklearn.preprocessing import LabelEncoder

model = load_model('static/file/best_trained_model.keras')

# Load and preprocess the dataset
df = pd.read_csv("static/file/Tweets(1).csv")
review_df = df[['text', 'airline_sentiment']]  # Select only the text and sentiment columns

# Preprocess sentiment labels
labels = review_df['airline_sentiment'] # Get the sentiment labels
text_data = review_df['text'] # Get the text data

label_encoder = LabelEncoder() # Initialize a label encoder
encoded_labels = label_encoder.fit_transform(labels) # Encode the sentiment labels

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text]) # Tokenize the text
    padded_sequence = pad_sequences(sequence, maxlen=200) # Pad the sequence
    prediction = model.predict(padded_sequence) # Make a prediction
    predicted_label = prediction.argmax(axis=-1)[0] # Get the predicted label
    return label_encoder.inverse_transform([predicted_label])[0] # Convert the label to the original sentiment
