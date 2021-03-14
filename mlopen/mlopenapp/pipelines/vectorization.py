from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def bag_of_words(corpus):
    bow = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    bow.fit(corpus)
    return bow

def tf_idf(corpus):
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    tfidf.fit(corpus)
    return tfidf
