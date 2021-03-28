import os
import numpy as np
from sklearn.datasets import make_blobs
import gensim
import torchtext.vocab as vocab
from tqdm import tqdm_notebook
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim import corpora

import torch

from NeuralNetworkFeedForward.ff_model import FeedforwardNeuralNetModel

from input import text_files_input as tfi
import text_preprocessing as tpp



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
    W2V_MIN_COUNT = 2  # Collect corpus for training word embeddings
    documents = [tokenize(_text) for _text in np.array(train.summary)]
    documents = documents + [tokenize(_text) for _text in
                             np.array(train.title)]  # Train Word Embeddings and save
    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, window=W2V_WINDOW,
                                                min_count=W2V_MIN_COUNT)
    w2v_model.build_vocab(documents)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)  # Train Word Embeddings
    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
    w2v_model.save('embeddings.txt')


def blob_label(y, label, loc):  # assign labels
    target = np.copy(y)
    for l in loc:
        target[y == l] = label
    return target


def ff_model(df, col_name):
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
            print(row[col_name])
            print(row['sentiment'])
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

# Function to make bow vector to be used as input to network
def make_bow_vector(review_dict, sentence, device, num_labels=2):
    VOCAB_SIZE = len(review_dict)
    NUM_LABELS = num_labels
    vec = torch.zeros(VOCAB_SIZE, dtype=torch.float64, device=device)
    for word in sentence:
        vec[review_dict.token2id[word]] += 1
    return vec.view(1, -1).float()


def make_target(label, device):
    if label == 0:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 1:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([2], dtype=torch.long, device=device)


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


x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
print(x_train)
print(y_train)

df_train = tpp.process_text_df(tfi.prepare_data(train_paths, train_sentiments), 'text')
df_test = tpp.process_text_df(tfi.prepare_data(test_paths, test_sentiments), 'text')
ff_model(df_train, 'text')
