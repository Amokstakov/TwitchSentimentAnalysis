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
import os
import sys
import spacy
import numpy as np
import pandas as pd
# import tensorflow as tf
from spacy.lang.en.stop_words import STOP_WORDS

from clean import contractions

# Bring in our Spacy Language model
nlp = spacy.load('en_core_web_md')

# import the data
df = pd.read_csv("./../Python/Data/training.1600000.processed.noemoticon.csv",
                 encoding='latin1', header=None)

# Switch the tweet columns with the sentiment columns
df = df[[5, 0]]
df.columns = ['Tweets', 'Sentiment']

# # Let's Continnue to Examine it

# # Look at the len of words in each Tweet
df['word_len'] = df['Tweets'].apply(lambda x: len(str(x).split()))


# Get average len of a word
def get_avrg_len(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len/len(words)


df['word_avrg_len'] = df['Tweets'].apply(lambda x: get_avrg_len(x))

# # Stop words
df['stop_word_len'] = df['Tweets'].apply(lambda x: len(
    [words for words in x.split() if words in STOP_WORDS]))

# Remove most frequently used words and least frequently used words

# Lets look at the most frequently used words
text = ' '.join(df['Tweets'])
text = text.split()
freq_ = pd.Series(text).value_counts()
Top_20 = freq_[:20]

# Lets look at the least freuqned used words
Least_20 = freq_[freq_.values == 1]


def contractions_replace(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    return x


def clean_text(x):
    x = x.lower()
    # remove all emails
    x = re.sub('([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)', "", x)

    # remove all @ first
    x = re.sub(r'@([A-Za-z0-9_]+)', "", x)

    # remove all # first
    x = re.sub(r'#([A-Za-z0-9_]+)', "", x)

    # remove and strip all retweets (RT)
    x = re.sub(r'\brt:\b', '', x).strip()

    # remove all websites
    # TODO: Figure out how it works for all possible website protocols
    x = re.sub(
        r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x)

    # clean and replace with contractions
    x = contractions_replace(x)

    # remove all numerical values
    x = re.sub(r'[0-9]+', "", x)

    # remove all special characters
    x = re.sub(r'[^\w ]+', ' ', x)

    # split aka tokenize our tweets
    x = x.split()

    # We are removed all the workds that are in our top 20
    x = [words for words in x if words not in Top_20]

    # We are rempoving all the words that in our rate 20
    x = [words for words in x if words not in Least_20]

    # remove all the words in our STOP_WORDS
    x = [words for words in x if words not in STOP_WORDS]

    return " ".join(x)


df = df[:5000]
print(df['Tweets'])
df['Tweets'] = df['Tweets'].apply(lambda x: clean_text(x))
print(df['Tweets'])
