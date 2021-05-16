import os
import numpy as np
import pandas
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
import torchtext.vocab as vocab
from tqdm import tqdm_notebook
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
from gensim import corpora
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from sklearn.metrics import mean_squared_error
from mlopenapp.utils import io_handler as io
from mlopenapp.utils import plotter


import torch

from mlopenapp.pipelines.NeuralNetworkFeedForward.ff_model import FeedforwardNeuralNetModel, Feedforward, LSTM_variable_input

from mlopenapp.pipelines.input import text_files_input as tfi
from mlopenapp.pipelines import text_preprocessing as tpp, vectorization as vct


ff_nn_bow_model = None


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


def word_2_vec(corpus):
    W2V_SIZE = 300
    W2V_WINDOW = 7
    W2V_EPOCH = 100
    W2V_MIN_COUNT = 2
    for c in corpus:
        print(c)
        if 'great' == c:
            print('OK GREAT IS HERE')
    w2v_model = gensim.models.word2vec.Word2Vec(corpus, window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT)
    print("most similar to 'evil : ",
          w2v_model.wv.most_similar('evil'))
    w2v_model.build_vocab(corpus)
    words = w2v_model.wv.key_to_index
    #print(words)
    vocab_size = len(words)
    print("Vocab size", vocab_size)  # Train Word Embeddings
    w2v_model.train(corpus, total_examples=len(corpus), epochs=W2V_EPOCH)
    print("most similar to 'evil : ",
          w2v_model.wv.most_similar('evil'))
    w2v_model.train(corpus, total_examples=len(corpus), epochs=W2V_EPOCH)
    print("most similar to 'evil : ",
          w2v_model.wv.most_similar('evil'))
    # w2v_model.save('embeddings.txt')
    return w2v_model


def get_doc2vec_model(corpus):
    card_docs = [TaggedDocument(doc, [i])
                 for i, doc in enumerate(corpus)]
    print(card_docs)
    model = Doc2Vec(vector_size=64, min_count=1, epochs=20)
    model = Doc2Vec(vector_size=64, window=2, min_count=1, workers=8, epochs=40)
    # build vocab
    model.build_vocab(card_docs)
    # train model
    model.train(card_docs, total_examples=model.corpus_count
                , epochs=model.epochs)
    return model


def vectorize_text(model, corpus):
    card2vec = [model.infer_vector(statement) for statement in corpus]
    vec_list = np.array(card2vec).tolist()
    print(vec_list)
    return vec_list


def blob_label(y, label, loc):  # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


def nnff_tfidf(df_train, df_test, xtest, arg):
    model = Feedforward(64, 2)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.eval()
    x_train = torch.FloatTensor(df_train['vectors'].tolist())
    x_test = torch.FloatTensor(df_test['vectors'].tolist())
    y_pred = model(x_train)
    y_train = torch.FloatTensor(df_train['sentiment'].tolist())
    y_test = torch.FloatTensor(df_test['sentiment'].tolist())
    before_train = criterion(y_pred.squeeze(), y_train)
    print('Test loss before training', before_train.item())

    model.train()
    epoch = 80
    for epoch in range(epoch):
        optimizer.zero_grad()  # Forward pass
        y_pred = model(x_train)  # Compute Loss
        loss = criterion(y_pred.squeeze(), y_train)

        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))  # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    x_test = torch.FloatTensor(xtest)
    y_test = torch.FloatTensor([0])
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss after Training', after_train.item())
    print(len(y_pred))
    print(len(y_test))
    for pred, real in zip(y_pred.squeeze(), y_test):
        print("Prediction is "+ str(pred) + ", real is " + str(real))
    return model

def ff_model(df, df_test, col_name, ff_nn_bow_model):
    # Use cuda if present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device available for running: ")
    print(device)
    # Make the dictionary without padding for the basic models
    review_dict = make_dict(df[col_name], padding=False)
    VOCAB_SIZE = len(review_dict)

    input_dim = VOCAB_SIZE
    hidden_dim = 500
    output_dim = 3
    num_epochs = 100

    ff_nn_bow_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    ff_nn_bow_model.to(device)
    ff_nn_bow_model.review_dict = review_dict
    ff_nn_bow_model.device = device

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ff_nn_bow_model.parameters(), lr=0.001)
    losses = []
    iter = 0

    # Start training
    for epoch in range(num_epochs):
        if (epoch + 1) % 25 == 0:
            print("Epoch completed: " + str(epoch + 1))
        train_loss = 0
        for index, row in df.iterrows():
            # Clearing the accumulated gradients
            optimizer.zero_grad()
            # Make the bag of words vector for stemmed tokens
            bow_vec = make_bow_vector(review_dict, row[col_name], device)

            # Forward pass to get output
            probs = ff_nn_bow_model(bow_vec)

            # Get the target label
            target = make_target(row['sentiment'], device)

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(probs, target)
            # Accumulating the loss over time
            train_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
        losses.append(str((epoch + 1)) + "," + str(train_loss / len(df)))
        train_loss = 0
    print(losses)
    test_model(df_test, col_name, ff_nn_bow_model)


def test_model(df_test, col_name, ff_nn_bow_model):
    bow_ff_nn_predictions = []
    original_lables_ff_bow = []
    with torch.no_grad():
        for index, row in df_test.iterrows():
            bow_vec = make_bow_vector(ff_nn_bow_model.review_dict, row[col_name], ff_nn_bow_model.device)
            probs = ff_nn_bow_model(bow_vec)
            bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables_ff_bow.append(make_target(row['sentiment'], ff_nn_bow_model.device).cpu().numpy()[0])
    print(classification_report(original_lables_ff_bow,bow_ff_nn_predictions))


# Function to make bow vector to be used as input to network
def make_bow_vector(review_dict, sentence, device, num_labels=2):
    VOCAB_SIZE = len(review_dict)
    NUM_LABELS = num_labels
    vec = torch.zeros(VOCAB_SIZE, dtype=torch.float64, device=device)
    for word in sentence:
        print(word)
        if word in review_dict.token2id:
            vec[review_dict.token2id[word]] += 1
    return vec.view(1, -1).float()


def make_target(label, device):
    if label == 0:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 1:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([2], dtype=torch.long, device=device)


# LTSM FEEDFORWARD FUNCTIONS AHEAD


class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], \
               self.X[idx][1]


def train_model(train_dl, val_dl, model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        print("Epoch " + str(i))
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            print("TRAIN DL HAS")
            print(y)
            print(x)
            print(l)
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            print("TRAIN PRED IS")
            print(y_pred)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))


def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


# Function to return the dictionary either with padding word or without padding
def make_dict(corpus, padding=True):
    if padding:
        print("Dictionary with padded token added")
        review_dict = corpora.Dictionary([['pad']])
        review_dict.add_documents(corpus)
    else:
        print("Dictionary without padding")
        review_dict = corpora.Dictionary(corpus)
    return review_dict


def encode_sentence(text, vocab2index, tokenized=True, N=70):
    if tokenized:
        tokens = text
    else:
        tokens = tpp.process_text(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokens])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


def get_vocab(df, arg):
    df[arg] = df[arg].apply(lambda x: tpp.process_text(x))
    df[arg + '_length'] = df[arg].apply(lambda x: len(x))
    counts = Counter()
    for index, row in df.iterrows():
        #print("row text is")
        #print(row[arg])
        counts.update(row[arg])
    #print("num_words before:", len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    #print("num_words after:", len(counts.keys()))

    # creating vocabulary
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    df['encoded'] = df[arg].apply(
        lambda x: np.array(encode_sentence(x, vocab2index)))
    #print(df.head())
    #print(vocab2index)
    return df, words, vocab2index


def prepare_datasets(df_c, arg, train=True, words=None, vocab=None):
    if words is None or vocab is None:
        df, words, vocab = get_vocab(df_c, arg)
    else:
        df = df_c.copy()
        df[arg] = df[arg].apply(lambda x: tpp.process_text(x))
        df[arg + '_length'] = df[arg].apply(lambda x: len(x))
        df['encoded'] = df[arg].apply(
            lambda x: np.array(encode_sentence(x, vocab)))

    # print(Counter(df['sentiment']))
    batch_size = 5000
    vocab_size = len(words)

    X = list(df['encoded'])
    y = list(df['sentiment']) if train else df['order']

    if train:
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        train_ds = ReviewsDataset(X_train, y_train)
        valid_ds = ReviewsDataset(X_valid, y_valid)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(valid_ds, batch_size=batch_size)
        return train_ds, train_dl, valid_ds, test_dl, words, vocab, vocab_size
    else:
        X_train = X
        y_train = y
        train_ds = ReviewsDataset(X_train, y_train)
        #print("REVIEWS DATASET IS")
        #print(train_ds)
        #print("DATALOADER IS")
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        #print(train_dl)
        return train_dl


def train_lstm_model(df, arg):
    train_ds, train_dl, valid_ds, test_dl, words, vocab, vocab_size = prepare_datasets(df, arg, True)
    model = LSTM_variable_input(vocab_size, 50, 50)
    batch_size = 5000
    vocab_size = len(words)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)
    train_model(train_dl, val_dl, model, epochs=80, lr=0.1)
    return model, vocab, words, batch_size


def train(df_train, arg):
    df_train = tfi.prepare_data(train_paths, train_sentiments)
    l_model, vocab, words, batch_size = train_lstm_model(df_train, 'text')
    models = [(l_model, "lstm_model")]
    args = [(vocab, "lstm_vocab"), (words, "lstm_words")]
    io.save_pipeline(models, args, os.path.basename(__file__))
    return l_model, vocab, words, batch_size


def run_pipeline(input, model, args):
    df = pandas.DataFrame(input, columns=['text'])
    preds = {'data': [], 'columns': [], 'graphs': None}
    arg = 'text'
    print(df)
    print(arg)
    print(df[arg])
    df['order'] = [i for i in range(0, len(df[arg]))]
    print(df)
    dl = prepare_datasets(df, arg, False, args['lstm_words'], args['lstm_vocab'])
    for x, y, l in dl:
        print(x)
        print(y)
        print(l)
        print("Prediction is")
        y_hat = model(x, l)
        pred = torch.max(y_hat, 1)[1].numpy()
        print(pred)
        pos = 0
        for p, i in zip(pred, y):
            print("Prediction is")
            preds['data'].append([df[arg].iat[int(i)], str(p)])
            if p == 1:
                pos += 1
            print(p)
        preds['columns'] = ['Statement', 'Sentiment']
        preds['graphs'] = plotter.plotlify_pie(
            {'Positive': pos, 'Negative': len(preds['data']) - pos},
            "Number of Positive and Negative Reviews")
        print(preds['graphs'])
        return preds



# lstm model implementation
"""
df_train = tfi.prepare_data(train_paths, train_sentiments)
l_model, vocab, words, batch_size = train(df_train, 'text')
example = [["This movie was one very pleasant surprise. The awesome was incredible. Must see!", 0],
            ["Worst thing I've ever seen in my life, avoid at all costs!", 1],
            ["I liked it, I don't care what everybody says, it was one of my favorites.", 2],
            ["Please, don't watch this! It's a total waste of time!", 3],
            ["I thought I would not like this, but it turned out to be pretty good!", 4],
            ["Who would have thought that such an expensive play would be so low quality", 5],
            ["I would wait bad worst to rent this. It does not justify a full price ticket", 6],
            ["Started awful, but it became too slow and unimaginative in the end", 7],
            ["A very bad movie, awful visuals, horrible sound - I hated it.", 8],
            ["I was sceptical at first, but this movie won me over - a great documentary!", 9]
           ]
df_ex = pandas.DataFrame(example, columns=['text', 'sentiment'])
print("EXAMPLE DF IS")
print(df_ex)
run_pipeline(df_ex, 'text', l_model, {'lstm_words': words, 'lstm_vocab': vocab})

"""


# word2vec model
"""
df_train = tpp.process_text_df(tfi.prepare_data(train_paths, train_sentiments), 'text')
df_test = tpp.process_text_df(tfi.prepare_data(test_paths, test_sentiments), 'text')
w2vm = word_2_vec(df_train['text'].tolist())
print(w2vm.wv['movie'])
print(w2vm.wv["resentful"])
vcts = []
"""

# feedforward implementation
"""

df_train = tpp.process_text_df(tfi.prepare_data(train_paths, train_sentiments), 'text')
df_test = tpp.process_text_df(tfi.prepare_data(test_paths, test_sentiments), 'text')

vectorizer = get_doc2vec_model(df_train['text'].tolist())
vec_list = vectorize_text(vectorizer, df_train['text'].tolist())
xtest = vectorize_text(vectorizer, [['bad', 'awful', 'sad', 'wasted']])
df_train['vectors'] = vec_list
vec_test_list = vectorize_text(vectorizer, df_test['text'].tolist())
df_test['vectors'] = vec_test_list
print(df_train)

#word_2_vec(df_train['text'].tolist())

ffmodel = nnff_tfidf(df_train, df_test, xtest, 'text')

#ff_model(df_train, df_test, 'text', ff_nn_bow_model)
"""
