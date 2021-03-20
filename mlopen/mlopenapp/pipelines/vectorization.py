from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def bag_of_words(corpus):
    bow = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    bow.fit(corpus)
    return bow

def fit_tf_idf(corpus):
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    tfidf.fit(corpus)
    return tfidf

def tf_idf(corpus):
    tfidf = fit_tf_idf(corpus)
    tfidf_mtx = tfidf.transform(corpus)
    return tfidf, tfidf_mtx
