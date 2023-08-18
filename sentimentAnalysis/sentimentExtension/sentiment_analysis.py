import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# Load your data and preprocess it
file_path = r"C:\Users\Einzcare\Desktop\django_project\Sentiment Analysis\data.csv"
column_names = ["Column0", "Column1", "Column2", "Column3", "Column4", "Column5"]
df = pd.read_csv(file_path, names=column_names)
review_df = df[['Column5', 'Column0']]

# Preprocess sentiment labels
sentiment_label = review_df.Column0.factorize()

# Preprocess text data using Tokenizer
sentence = review_df.Column5.values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(sentence)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

# Recreate the model architecture
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the weights from the saved checkpoint
model.load_weights('static/file/model_weights_epoch_05.h5')

# Now you can continue with the rest of your code
checkpoint = ModelCheckpoint('model_weights_epoch_{epoch:02d}.h5', save_weights_only=True)

history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs=5, initial_epoch=5, batch_size=64, callbacks=[checkpoint])
model.save('trained_model.keras')

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])

test_sentence1 = "I enjoyed my journey on this flight."
predict_sentiment(test_sentence1)

test_sentence2 = "This is the worst flight experience of my life!"
predict_sentiment(test_sentence2)
