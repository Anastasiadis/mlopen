import spacy
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from django.conf import settings
import matplotlib.pyplot as plt
import contractions
"""
text = 
Dave watched as the forest burned up on the hill,
only a few miles from his house. The car had
been hastily packed and Marta was inside trying to round
up the last of the pets. "Where could she be?" he wondered
as he continued to wait for Marta to appear with the pets.


nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Tokenization
token_list = [token for token in doc]

# Stopword removal
filtered_tokens = [token for token in doc if not token.is_stop]
print(filtered_tokens)

# Lemmatization
lemmas = [
    f"Token: {token}, lemma: {token.lemma_}" for token in filtered_tokens
]

# Vectorization
"""

tools_init = [False]

def initialize_tools(ti=tools_init):
    if not ti[0]:
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')
        ti[0] = True


# function to convert nltk tag to wordnet tag
def nltk_to_wn(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def to_lower(text):
    return text.lower()


def rm_word_repetition(text):
    return re.sub(r'(.)\1+', r'\1\1', text)


def rm_punctuation_repetition(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', "", text)


def replace_contractions(text):
    return contractions.fix(text)


def tokenize_text(text):
    return word_tokenize(text)


def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def only_alpha(tokens):
    tokens = [token for token in tokens if token.isalpha()]
    return tokens


def lemmatize(tokens, verbose=False):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.pos_tag(tokens)
    if verbose:
        print(tokens)
    for i, token in enumerate(tokens):
        tokens[i] = lemmatizer.lemmatize(token[0], nltk_to_wn(token[1]))
    return tokens


def process_text(text, verbose=False):
    if verbose:
        print("Pre Word processing text: {}".format(text))
    initialize_tools()
    text = to_lower(text)
    text = rm_word_repetition(text)
    text = rm_punctuation_repetition(text)
    text = replace_contractions(text)
    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens)
    tokens = only_alpha(tokens)
    tokens = lemmatize(tokens)
    if verbose:
        print("Post Word processing text: {}".format(str(tokens)))
    return tokens


def process_text_df(df, *args):
    for arg in args:
        for i, text in df[arg].items():
            tokens = process_text(text)
            df[arg].at[i] = tokens
    return df


#process_text("That's NOT what I'm saying or said!!! I sould not be telling you this, but I run and while running I fell..", True)