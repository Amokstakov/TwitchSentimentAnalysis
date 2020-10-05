"""
We need to import the csv file
We need to import our GloVe embeddings
We need to create and tune our deep neural network
We need to create SVM and other methods 
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Bidirectional, GlobalAveragePooling1D, concatenate, LeakyReLU, LSTM
from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D


df = pd.read_csv('clean_csv_4k.csv')
df = df.fillna('')

with open('test_file.pickle', 'rb') as handle:
    embeddings = pickle.load(handle)

# convert from series to a list
text = df['twitts'].tolist()

y = df['sentiment']

token = Tokenizer()
token.fit_on_texts(text)

vocab_size = len(token.word_index) + 1

encoded_text = token.texts_to_sequences(text)

# Pad the sequences
max_len = max([len(s.split()) for s in text])

X = pad_sequences(encoded_text, maxlen=max_len, padding='post')

# our task is to get the global vectors for our words
# create empty matrix with the proper size
word_vector_matrix = np.zeros((vocab_size, 200))

for word, index in token.word_index.items():
    vector = embeddings.get(word)
    # check if the word is not present in GloVe
    if vector is not None:
        word_vector_matrix[index] = vector
    else:
        print(word)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)

vec_size = 200

model = tf.keras.Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_len,
                    weights=[word_vector_matrix], trainable=False))
model.add(Conv1D(64, 8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
# model = tf.keras.Sequential()
# model.add(Embedding(vocab_size, vec_size, input_length=max_len,
                    # weights=[word_vector_matrix], trainable=False))
# model.add(SpatialDropout1D(0.5))
# #We can toy with filter and kernel values
# model.add(Conv1D(64, 4, kernel_regularizer=regularizers.l2(0.00001))) 
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling1D(pool_size=2))
# #can change LSTM units
# model.add(Bidirectional(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
# model.add(SpatialDropout1D(0.5))
# model.add(Conv1D(64, 4, kernel_regularizer=regularizers.l2(0.00001))) 
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling1D(pool_size=2))
# #can change LSTM units
# model.add(Bidirectional(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
# model.add(SpatialDropout1D(0.5))
# model.add(Conv1D(64, 4, kernel_regularizer=regularizers.l2(0.00001))) 
# model.add(LeakyReLU(alpha=0.2))
# model.add(MaxPooling1D(pool_size=2))
# #can change LSTM units
# model.add(Bidirectional(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
# model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30,batch_size=32, validation_data=(x_test, y_test))


svm_pipeline = Pipeline([('svm',LinearSVC())])
svm_pipeline.fit(x_train, y_train)

svm_test_predictions = svm_pipeline.predict(x_test)

print(metrics.accuracy_score(y_test,svm_test_predictions))










