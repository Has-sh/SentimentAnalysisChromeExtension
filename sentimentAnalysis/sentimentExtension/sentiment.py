from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

model = load_model('static/file/trained_model_multiclass.h5')

df = pd.read_csv("static/file/Tweets(1).csv")
review_df = df[['text','airline_sentiment']]

sentiment_label = review_df.airline_sentiment.factorize()

sentence = review_df.text.values

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentence)

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = model.predict(tw)

    predicted_label_index = np.argmax(prediction)
    predicted_sentiment = sentiment_label[1][predicted_label_index]
    return predicted_sentiment