"""
Take 1.6 Million Tweets
These Tweets are already labelled. Data source: Kaggle:
    Kazanova sentimment analysis Twitter Data
First we will examine the tweets
Then we will perform data cleaning on each tweet, respectively:
    turn it all into lower case : done
    remove @usernames : done
    remove any potential URLS : done
    remove any potential Retweets : done
    remove stop words: done
    remove hashtags: doine 
    remove contractions (english text) : done 
    remove any emails : done
    tokenize the reminaing text done
    Use word embeddings: 
        - GloVe : Specifically Trainined Twitter GloVe embeddings
        - Pre-built Spacy
"""
# imports
import re
import sys
import spacy
import pickle
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
from textblob import TextBlob
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Flatten, Dropout, Embedding, Activation, Conv1D, MaxPooling1D, GlobalMaxPooling1D

from clean import contractions

nlp = spacy.load('en_core_web_md')

df = pd.read_csv('../Python/Data/twitter-data-master/twitter4000.csv'

# Find most frequent and rarest word ocrruences
text = ' '.join(df['twitts'])
text = text.split()
freq_ = pd.Series(text).value_counts()
Top_10 = freq_[:10]
Least_freq = freq_[freq_.values == 1]

# Clean the data
def get_cleat_text(text):
    if type(text) is str:
        text = text.lower()
        # find and replace all emails
        text = re.sub(
        "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", '', text)
        # remove all @ first
        text = re.sub(r'@([A-Za-z0-9_]+)', "", text)
        # # remove and strip all retweets (RT)
        text = re.sub(r'\brt:\b', '', text).strip()
        # find and replace all websites
        text = re.sub(
            r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', text)
        # find and replace all non-alpha numerical valu
        text = re.sub(r'[^A-Z a-z]+', '', text)
        # #Remove accented characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text
    else:
        return text

df['twitts'] = df['twitts'].apply(lambda x: get_cleat_text(x))

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

with open('test_file.pickle', 'rb') as handle:
    data_test = pickle.load(handle)

# our task is to get the global vectors for our words
# create empty matrix with the proper size
word_vector_matrix = np.zeros((vocab_size, 200))

for word, index in token.word_index.items():
    vector = data_test.get(word)
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

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))


