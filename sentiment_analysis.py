import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences       
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

# Load your data and preprocess it
file_path = 'static/file/Tweets(1).csv'
column_names = ["tweet_id", "airline_sentiment", "airline_sentiment_confidence", "negativereason", "negativereason_confidence", "airline", "airline_sentiment_gold", "name", "negativereason_gold", "retweet_count", "text", "tweet_coord", "tweet_created", "tweet_location", "user_timezone"]
df = pd.read_csv(file_path, names=column_names, header=0) # Load the dataset

review_df = df[['text', 'airline_sentiment']]  # Select only the text and sentiment columns

# Preprocess sentiment labels
labels = review_df['airline_sentiment'] # Get the sentiment labels
text_data = review_df['text'] # Get the text data

label_encoder = LabelEncoder() # Initialize a label encoder
encoded_labels = label_encoder.fit_transform(labels) # Encode the sentiment labels

# Preprocess text data using Tokenizer
tokenizer = Tokenizer() # Initialize a tokenizer
tokenizer.fit_on_texts(text_data) # Fit the tokenizer on the text data
vocab_size = len(tokenizer.word_index) + 1 # Get the vocabulary size 
encoded_docs = tokenizer.texts_to_sequences(text_data) # Encode the text data
padded_sequences = pad_sequences(encoded_docs, maxlen=200) # Pad the sequences

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42) # Split the data into training and testing sets

# Define the model architecture
def create_model(embedding_vector_length=32, lstm_units=100, dropout_rate=0.2, spatial_dropout_rate=0.2):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
    model.add(SpatialDropout1D(spatial_dropout_rate))
    model.add(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    model.add(Dropout(dropout_rate))
    model.add(Dense(3, activation='softmax'))  # Adjust the number of units to match the number of classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Define hyperparameters to tune
# param_grid = {
#     'embedding_vector_length': [32, 64],
#     'lstm_units': [50, 100],
#     'dropout_rate': [0.2, 0.3],
#     'spatial_dropout_rate': [0.2, 0.3],
#     'batch_size': [64, 128],
#     'epochs': [5, 10]
# }
#Best validation accuracy: 0.812158465385437 with parameters: {'embedding_vector_length': 64, 'lstm_units': 100, 'dropout_rate': 0.3, 'spatial_dropout_rate': 0.2, 'batch_size': 64, 'epochs': 10}
param_grid = {
    'embedding_vector_length': [64],
    'lstm_units': [100],
    'dropout_rate': [0.3],
    'spatial_dropout_rate': [0.2],
    'batch_size': [64],
    'epochs': [10]
}
# Perform manual hyperparameter tuning
best_model = None
best_accuracy = 0
best_params = {}

for embedding_vector_length in param_grid['embedding_vector_length']:
    for lstm_units in param_grid['lstm_units']:
        for dropout_rate in param_grid['dropout_rate']:
            for spatial_dropout_rate in param_grid['spatial_dropout_rate']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        model = create_model(embedding_vector_length, lstm_units, dropout_rate, spatial_dropout_rate)
                        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
                        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=early_stopping)
                        
                        val_accuracy = max(history.history['val_accuracy'])
                        
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            best_model = model
                            best_params = {
                                'embedding_vector_length': embedding_vector_length,
                                'lstm_units': lstm_units,
                                'dropout_rate': dropout_rate,
                                'spatial_dropout_rate': spatial_dropout_rate,
                                'batch_size': batch_size,
                                'epochs': epochs
                            }

print(f"Best validation accuracy: {best_accuracy} with parameters: {best_params}")

#Best validation accuracy: 0.812158465385437 with parameters: {'embedding_vector_length': 64, 'lstm_units': 100, 'dropout_rate': 0.3, 'spatial_dropout_rate': 0.2, 'batch_size': 64, 'epochs': 10}
# Save the best model
best_model.save('static/file/best_trained_model.keras')

# Define the prediction function
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text]) # Tokenize the text
    padded_sequence = pad_sequences(sequence, maxlen=200) # Pad the sequence
    prediction = model.predict(padded_sequence) # Make a prediction
    predicted_label = prediction.argmax(axis=-1)[0] # Get the predicted label
    return label_encoder.inverse_transform([predicted_label])[0] # Convert the label to the original sentiment

# Test the prediction function
test_sentence1 = "I enjoyed my journey on this flight."
print("Prediction for test_sentence1:", predict_sentiment(test_sentence1))

test_sentence2 = "This is the worst flight experience of my life!"
print("Prediction for test_sentence2:", predict_sentiment(test_sentence2))

test_sentence3 = "Middle seat on a red eye. Such a noob maneuver"
print("Prediction for test_sentence3:", predict_sentiment(test_sentence3))