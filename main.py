import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from tensorflow import keras
from sklearn.model_selection import train_test_split

#get data
reddit_data = pd.read_csv('Reddit_Data.csv')
twitter_data = pd.read_csv('Twitter_Data.csv')

reddit_test, reddit_train = train_test_split(reddit_data, test_size=0.1)
twitter_test, twitter_train = train_test_split(twitter_data, test_size=0.1)

#merge data for training only and drop the index

train_data = pd.concat([reddit_train["clean_comment"], twitter_train["clean_text"]], ignore_index=True)
train_labels = pd.concat([reddit_train["category"], twitter_train["category"]], ignore_index=True)
train_data = train_data.astype(str)


twitter_test_labels = twitter_test["category"]
reddit_test_labels = reddit_test["category"]
twitter_test_data = twitter_test["clean_text"]
reddit_test_data = reddit_test["clean_comment"]

#create tokenizer


tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index

#convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_data)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=100)

#convert labels to one-hot encoding
train_labels = pd.get_dummies(train_labels)

#create model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 128, input_length=100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_padded, train_labels, epochs=20)

#test model
reddit_test_data = reddit_test_data.astype(str)
reddit_test_sequences = tokenizer.texts_to_sequences(reddit_test_data)
reddit_test_padded = pad_sequences(reddit_test_sequences, padding='post', maxlen=100)

twitter_test_data = twitter_test_data.astype(str)
twitter_test_sequences = tokenizer.texts_to_sequences(twitter_test_data)
twitter_test_padded = pad_sequences(twitter_test_sequences, padding='post', maxlen=100)

reddit_test_labels = pd.get_dummies(reddit_test_labels)
twitter_test_labels = pd.get_dummies(twitter_test_labels)

print(model.evaluate(reddit_test_padded, reddit_test_labels))
print(model.evaluate(twitter_test_padded, twitter_test_labels))

model.save('model.h5')

test_sentences = ["I hate the new iPhone"]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=100)

print(np.argmax(model.predict(test_padded)))

