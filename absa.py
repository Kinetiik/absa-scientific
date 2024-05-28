from datasets import load_dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
dataset = load_dataset("jakartaresearch/semeval-absa", "laptop")

reviews = dataset['train']['text']
labels = dataset['train']['aspects']

from_aspect = []
to_aspect = []
sentiment = []
for i in labels:
    sentiment.append(i['polarity'])
    from_aspect.append(i['from'])
    to_aspect.append(i['to'])


def sentiment_to_int(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return 2
    elif sentiment == 'neutral':
        return 3
    else:
        return 0
    
# Encode sentiments to one-hot encoding. Assuming 3 classes: positive, negative, neutral

sentiment_labels = []
aspect_labels = []

for i in range(len(from_aspect)):
    aspect_label = np.zeros(100)
    sentiment_label = np.full((100,), -1)
    for index, (j, k) in enumerate(zip(from_aspect[i], to_aspect[i])):
        aspect_label[j:k] = 1
        sentiment_label[j:k] = sentiment_to_int(sentiment[i][index])
    
    aspect_labels.append(aspect_label)
    sentiment_labels.append(sentiment_label)

aspect_labels = np.array(aspect_labels)
sentiment_labels = np.array(sentiment_labels)



#encode sentiment labels to one-hot encoding
sentiment_labels = np.eye(4)[sentiment_labels]

# Tokenize and pad the reviews
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded_reviews = pad_sequences(sequences, maxlen=100)

# Build Aspect Extraction Model
aspect_input = Input(shape=(100,))
aspect_embedding = Embedding(input_dim=10000, output_dim=128, input_length=100)(aspect_input)
aspect_lstm1 = Bidirectional(LSTM(64, return_sequences=True))(aspect_embedding)
aspect_lstm2 = Bidirectional(LSTM(64, return_sequences=True))(aspect_lstm1)
aspect_output = TimeDistributed(Dense(1, activation='sigmoid'))(aspect_lstm2)

aspect_model = Model(inputs=aspect_input, outputs=aspect_output)
aspect_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build Sentiment Classification Model
sentiment_input = Input(shape=(100,))
sentiment_embedding = Embedding(input_dim=10000, output_dim=128, input_length=100)(sentiment_input)
sentiment_lstm1 = Bidirectional(LSTM(64, return_sequences=True))(sentiment_embedding)
sentiment_lstm2 = Bidirectional(LSTM(64, return_sequences=True))(sentiment_lstm1)
sentiment_output = TimeDistributed(Dense(4, activation='softmax'))(sentiment_lstm2) # Assuming 3 classes: positive, negative, neutral

sentiment_model = Model(inputs=sentiment_input, outputs=sentiment_output)
sentiment_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the models
aspect_model.fit(padded_reviews, aspect_labels, epochs=50, batch_size=32)
sentiment_model.fit(padded_reviews,sentiment_labels, epochs=20, batch_size=32)

# Evaluate the models
aspect_model.evaluate(padded_reviews, aspect_labels)
sentiment_model.evaluate(padded_reviews, sentiment_labels)

# Predict on a dummy review
dummy_review = "The battery life is great but the screen is too small"
dummy_review_seq = tokenizer.texts_to_sequences([dummy_review])
dummy_review_padded = pad_sequences(dummy_review_seq, maxlen=100)

aspect_prediction = aspect_model.predict(dummy_review_padded)
sentiment_prediction = sentiment_model.predict(dummy_review_padded)

def interpret_prediction(aspect_prediction, sentiment_prediction, input_text):
    # All aspects above 0.5 are considered present
    aspect_threshold = 0.4
    sentiment_predictions = np.argmax(sentiment_prediction, axis=-1)
    
    # Use tokenizer to get the words from the input text
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    input_text = input_text[0]
    input_text = [index_word[i] for i in input_text if i != 0]
    
    aspect_words = []
    sentiment_aspects = []
    for i in range(len(input_text)):
        if aspect_prediction[0][i] >= aspect_threshold:
            aspect_words.append(input_text[i])
            sentiment_aspects.append(sentiment_predictions[0][i])
        
    return aspect_words, sentiment_aspects

aspect, sentiment = interpret_prediction(aspect_prediction, sentiment_prediction, dummy_review_seq)
print(f"Predicted aspect: {aspect}")
print(f"Predicted sentiment: {sentiment}")
