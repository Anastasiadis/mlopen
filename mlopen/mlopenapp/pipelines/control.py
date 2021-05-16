import os
from mlopenapp.pipelines import vectorization as vct,\
    text_preprocessing as tpp, \
    logistic_regression as lr, \
    metrics as mtr
from mlopenapp.utils import io_handler as io
from mlopenapp.utils import plotter, data_handler

# These will be replaced by user input
train_paths = [
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/train/pos/'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/train/neg/')
]

train_sentiments = [1, 0]

test_paths = [
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/test/pos/'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/test/neg/')
]

test_sentiments = [1, 0]


def train(df_train,  df_test, *args):
    """
    Creates a model based on a train and a test dataframes, and calculates model metrics
    """
    print("Preparing Data. . .")
    tfidf = {}
    vector = {}
    for arg in args:
        corpus = df_train[arg].tolist()
        tfidf[arg] = vct.fit_tf_idf(corpus)
        vector[arg] = vct.tf_idf(corpus, tfidf[arg])
        # TODO: add pos/neg frequencies method
        # freqs = frq.build_freqs(corpus, df['sentiment'].tolist())
        # x_pn = [frq.statement_to_freq(txt, freqs) for txt in corpus]
        # x_posneg = frq.get_posneg(corpus, freqs)
        s_a_model = lr.log_reg(vector[arg], df_train['sentiment'].tolist())
        test_corpus = df_test[arg].tolist()
        test_sentiment = df_test['sentiment'].tolist()
        mtr.get_model_metrics(tfidf[arg], s_a_model, test_corpus, test_sentiment, True)
        models = [(s_a_model, "logreg_model")]
        args = [(tfidf[arg], "tfidf_vect")]
        io.save_pipeline(models, args, os.path.basename(__file__))
        return tfidf[arg], s_a_model


def run_pipeline(input, model, args):
    """
    Predicts the sentiment of a list of text statements
    """
    preds = {'data': [], 'columns': [], 'graphs': None}
    pos = 0
    for statement in input:
        temp = predict_text(statement, args['tfidf'], model, False)
        preds['data'].append([str(temp[0]), str(temp[1])])
        if temp[1] == 1:
            pos += 1
    preds['columns'] = ['Statement', 'Sentiment']
    preds['graphs'] = plotter.plotlify_pie({'Positive': pos, 'Negative': len(preds['data']) - pos}, "Number of Positive and Negative Reviews")
    print(preds['graphs'])
    return preds


def predict_text(text, tfidf, model, processed=False):
    """
    Predicts the sentiment of a single text statement
    """
    original = text
    if not processed:
        text = tpp.process_text(text)
    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)
    return (original, prediction[0])


#df_train = tpp.process_text_df(tfi.prepare_data(train_paths, train_sentiments), 'text')
#df_test = tpp.process_text_df(tfi.prepare_data(test_paths, test_sentiments), 'text')
#model_tup = train(df_train, df_test, 'text')

