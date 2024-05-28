from datasets import load_dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional, Conv1D, GlobalMaxPooling1D, MultiHeadAttention, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from keras.optimizers import Adam

# Load dataset
dataset = load_dataset("jakartaresearch/semeval-absa", "laptop")

reviews = dataset['train']['text']
labels = dataset['train']['aspects']

#retrieve test data
test_reviews = dataset['validation']['text']
test_labels = dataset['validation']['aspects']


def sentiment_to_int(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return 2
    elif sentiment == 'neutral':
        return 3
    else:
        return 0

def convert_data(reviews, labels,tokenizer = None, embedding_matrix = None):
    from_aspect = []
    to_aspect = []
    sentiment = []
    for i in labels:
        sentiment.append(i['polarity'])
        from_aspect.append(i['from'])
        to_aspect.append(i['to'])

    if tokenizer is None:
    # Tokenize and pad the reviews
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(reviews)
        sequences = tokenizer.texts_to_sequences(reviews)
        padded_reviews = pad_sequences(sequences, maxlen=100)
        word_index = tokenizer.word_index
    else:
        sequences = tokenizer.texts_to_sequences(reviews)
        padded_reviews = pad_sequences(sequences, maxlen=100)
        word_index = tokenizer.word_index

    # Prepare labels
    aspect_labels = []
    sentiment_labels = []

    for i, review in enumerate(reviews):
        sequence = sequences[i]
        aspect_label = np.zeros(100)
        sentiment_label = np.full(100, 0)
        for j, k, polarity in zip(from_aspect[i], to_aspect[i], sentiment[i]):
            # Find the corresponding token indices for aspect span
            tokens_in_aspect = tokenizer.texts_to_sequences([review[j:k]])[0]
            if not tokens_in_aspect:
                continue  # Skip if no tokens found in aspect span
            try:
                token_start_index = sequence.index(tokens_in_aspect[0])
                token_end_index = token_start_index + len(tokens_in_aspect)
                aspect_label[token_start_index:token_end_index] = 1
                sentiment_label[token_start_index:token_end_index] = sentiment_to_int(polarity)
            except ValueError:
                continue  # Skip if token not found in sequence
        
        aspect_labels.append(aspect_label)
        sentiment_labels.append(sentiment_label)
    

    aspect_labels = np.array(aspect_labels)
    sentiment_labels = np.array(sentiment_labels)

    # Encode sentiment labels to one-hot encoding
    sentiment_labels = np.eye(4)[sentiment_labels]

    if embedding_matrix is None:
        # Load pre-trained embeddings (e.g., GloVe)
        embedding_index = {}
        with open('glove.6B.100d.txt', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs

        embedding_dim = 100
        embedding_matrix = np.zeros((10000, embedding_dim))
        for word, i in tokenizer.word_index.items():
            if i < 10000:
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix = embedding_matrix
        embedding_dim = 100

    return padded_reviews, aspect_labels, sentiment_labels, tokenizer, embedding_matrix, embedding_dim


padded_reviews, aspect_labels, sentiment_labels, tokenizer, embedding_matrix, embedding_dim = convert_data(reviews, labels)

# Build Aspect Extraction Model
aspect_input = Input(shape=(100,))
aspect_embedding = Embedding(input_dim=10000, output_dim=embedding_dim, weights=[embedding_matrix], input_length=100, trainable=False)(aspect_input)
aspect_conv = Conv1D(filters=128, kernel_size=5, activation='relu')(aspect_embedding)
aspect_pool = GlobalMaxPooling1D()(aspect_conv)
aspect_lstm1 = Bidirectional(LSTM(128, return_sequences=True))(aspect_embedding)
aspect_lstm2 = Bidirectional(LSTM(128, return_sequences=True))(aspect_lstm1)
aspect_attention = MultiHeadAttention(num_heads=2, key_dim=128)(aspect_lstm2, aspect_lstm2)
aspect_output = TimeDistributed(Dense(1, activation='sigmoid'))(aspect_attention)

aspect_model = Model(inputs=aspect_input, outputs=aspect_output)
aspect_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Build Sentiment Classification Model
sentiment_input = Input(shape=(100,))
sentiment_embedding = Embedding(input_dim=10000, output_dim=embedding_dim, weights=[embedding_matrix], input_length=100, trainable=False)(sentiment_input)
sentiment_conv = Conv1D(filters=128, kernel_size=5, activation='relu')(sentiment_embedding)
sentiment_pool = GlobalMaxPooling1D()(sentiment_conv)
sentiment_lstm1 = Bidirectional(LSTM(128, return_sequences=True))(sentiment_embedding)
sentiment_lstm2 = Bidirectional(LSTM(128, return_sequences=True))(sentiment_lstm1)
sentiment_attention = MultiHeadAttention(num_heads=2, key_dim=128)(sentiment_lstm2, sentiment_lstm2)
sentiment_dropout = Dropout(0.25)(sentiment_attention)
sentiment_output = TimeDistributed(Dense(4, activation='softmax'))(sentiment_dropout)

sentiment_model = Model(inputs=sentiment_input, outputs=sentiment_output)
sentiment_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the models
# aspect_model.fit(padded_reviews, aspect_labels, epochs=25, batch_size=256)
# sentiment_model.fit(padded_reviews, sentiment_labels, epochs=10, batch_size=256)

# # # # Save the models
# aspect_model.save('aspect_model.h5')
# sentiment_model.save('sentiment_model.h5')

aspect_model.load_weights('aspect_model.h5')
sentiment_model.load_weights('sentiment_model.h5')

test_padded_reviews, test_aspect_labels, test_sentiment_labels, _, _, _ = convert_data(test_reviews, test_labels, tokenizer, embedding_matrix)

# Evaluate the models
aspect_loss, aspect_accuracy = aspect_model.evaluate(test_padded_reviews, test_aspect_labels)
sentiment_loss, sentiment_accuracy = sentiment_model.evaluate(test_padded_reviews, test_sentiment_labels)

# Predict on a dummy review
dummy_review = test_reviews[4]
dummy_review_sequence = tokenizer.texts_to_sequences([dummy_review])
dummy_review_padded = pad_sequences(dummy_review_sequence, maxlen=100)

aspect_prediction = aspect_model.predict(dummy_review_padded)
sentiment_prediction = sentiment_model.predict(dummy_review_padded)

def interpret_prediction(aspect_prediction, sentiment_prediction, input_text, tokenizer):
    aspect = ""
    sentiment = ""
    input_tokens = tokenizer.texts_to_sequences([input_text])[0]
    padded_input_tokens = pad_sequences([input_tokens], maxlen=100)[0]
    for i, (aspect_prob, sentiment_prob) in enumerate(zip(aspect_prediction[0], sentiment_prediction[0])):
        if aspect_prob > 0.1 and i < len(padded_input_tokens) and padded_input_tokens[i] != 0:
            word = tokenizer.index_word.get(padded_input_tokens[i], '')
            aspect += word + " "
            sentiment += str(np.argmax(sentiment_prob)) + " "
    return aspect, sentiment

print(test_reviews[4])
print(aspect_prediction)
print(sentiment_prediction)


aspect, sentiment = interpret_prediction(aspect_prediction, sentiment_prediction, dummy_review, tokenizer)
print(f"Predicted aspect: {aspect}")
print(f"Predicted sentiment: {sentiment}")
