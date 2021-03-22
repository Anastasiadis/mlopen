import os
import csv
import pandas as pd
import prepare_data as pdt
import text_preprocessing as tpp


def make_model(*args):
    for arg in args:
        tfidf, vector, model = pdt.get_model(arg)
        print(tfidf)
        print(vector)
        print(model)
        return tfidf, vector, model


def make_prediction(input, tfidf, model, processed=False):
    for statement in input:
        print(statement)
        print(predict_text(statement, tfidf, model, processed))


def predict_text(text, tfidf, model, processed=False):
    if not processed:
        text = tpp.process_text(text)
    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)
    print(prediction)
    return prediction


s1 = ["Idiot worse evil bad puke sad.",
      "Best, the absolute greater beautiful movie!",
      "This movie was SICK!!! I was stunned by the plot twist in the end! I didn't expect it!",
      "Worst movie of all time. Awful actors, stupid plot, horrible ending.",
      "I didn't like this movie. It left me with a strange taste that made me feel weird. It wasn't scary at all.",
      "I guess the begining was fine, but it declined rapidly after 20 minutes. The movie isn't worth the ticket price - just watch it when it's free on television, or maybe not even then",
      "The only thing in this abyssymal, good for nothing movie, that the fact that it is short and you don't have to suffer for long."
      ]
model_tup = make_model('text')
make_prediction(s1, model_tup[0], model_tup[2])
