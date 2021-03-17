import os
import csv
import pandas as pd
import prepare_data as pdt
import text_preprocessing as tpp


def prepare_data(*args):
    for arg in args:
        tfidf, model = pdt.get_model(arg)
        print(tfidf)
        s1 = "I didn't like this movie. It left me with a strange taste that made me feel weird. It wasn't scary at all."
        #s1 = tpp.process_text(s1)
        #s1 = tfidf.transform(s1)
        #pred = model.predict(s1)
        #print(pred)




prepare_data('text')