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
import vectorization as vct,\
    text_preprocessing as tpp, \
    logistic_regression as lr, \
    frequencies as frq

"""
Function to calculate pos-neg frequencies
"""


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
        vector[arg] = vct.tf_idf(corpus)
        freqs = frq.build_freqs(df[arg].tolist(), df['sentiment'].tolist())
        x_posneg = frq.get_posneg(df[arg].tolist(), freqs)
        s_a_model = lr.log_reg(x_posneg, df['sentiment'].tolist())
        return vector[arg], s_a_model

