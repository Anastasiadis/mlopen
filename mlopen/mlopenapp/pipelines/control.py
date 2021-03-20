import os
import csv
import pandas as pd
import prepare_data as pdt
import text_preprocessing as tpp


def prepare_data(*args):
    for arg in args:
        tfidf, vector, model = pdt.get_model(arg)
        print(tfidf)
        print(vector)
        print(model)

        s1 = ["Idiot worse evil bad puke sad.",
              "Best, the absolute greater beautiful movie!",
              "This movie was SICK!!! I was stunned by the plot twist in the end! I didn't expect it!",
              "Worst movie of all time. Awful actors, stupid plot, horrible ending.",
              "I didn't like this movie. It left me with a strange taste that made me feel weird. It wasn't scary at all."
             ]
        for s in s1:
            print(s)
            print(predict_text(s, tfidf, model))
        #s1 = tpp.process_text(s1)
        #s1 = tfidf.transform(s1)
        #pred = model.predict(s1)
        #print(pred)


def predict_text(text, tfidf, model):
    text = tpp.process_text(text)
    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)
    print(prediction)
    if prediction == 1:
        return "Prediction is positive sentiment"
    else:
        return "Prediction is negative sentiment"


prepare_data('text')