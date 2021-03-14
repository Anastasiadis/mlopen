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

from input import text_files_input as tfi
import vectorization as vct, text_preprocessing as tpp


def get_model(*args, verbose=False):
    print("Preparing Data. . .")
    df = tfi.prepare_data()
    vector = {}
    for arg in args:
        print("For column " + str(arg) + ":")
        print("Processing Text. . .")
        df = tpp.process_text_df(df, arg)
        print("Create Corpus. . .")
        vector = {}
        corpus = []
        print("Create Vector. . .")
        for i, corp in df[arg].items():
            corpus.append(corp)
        vector[arg] = vct.bag_of_words(corpus)
    if True:
        print("Get model:")
        print(df)
        print(vector)


get_model('text')