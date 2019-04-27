# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

import os
import sys
sys.path.append(os.path.abspath('..'))
from util_data import get_wikipedia_data
from brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

from markov import get_bigram_probs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CUDA = torch.cuda.is_available()

def make_input(sequence):
    vec = torch.zeros(len(sequence))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

class Logistic(nn.Module):
    def __init__(self, V, D):
        super(Logistic, self).__init__()
        #self.W = nn.Linear(V, D)
        self.W = nn.Embedding(V, D)
        nn.init.uniform_(self.W.weight.data)
        
    def forward(self, input_vec):
        #to_feed = make_input(input_vec)
        out = self.W(input_vec)
        out = out - out.max()
        return F.log_softmax(out, dim=1)

if __name__ == '__main__':
    # load in the data
    # note: sentences are already converted to sequences of word indexes
    # note: you can limit the vocab size if you run out of memory
    sentences, word2idx = get_wikipedia_data(n_files=5, n_vocab=5000)
    #sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
    # sentences, word2idx = get_sentences_with_word2idx()

    # vocab size
    V = len(word2idx)
    print("Vocab size:", V)

    # we will also treat beginning of sentence and end of sentence as bigrams
    # START -> first word
    # last word -> END
    start_idx = word2idx['START']
    end_idx = word2idx['END']


    # a matrix where:
    # row = last word
    # col = current word
    # value at [row, col] = p(current word | last word)
    bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)


    # train a logistic model
    #W = np.random.randn(V, V) / np.sqrt(V)
    logistic = Logistic(V, V)
    optimizer = torch.optim.SGD(logistic.parameters(), lr=0.1)
    loss_function = nn.NLLLoss()

    if CUDA:
        logistic.to(torch.device("cuda"))

    losses = []
    epochs = 2
    lr = 1e-1

    # what is the loss if we set W = log(bigram_probs)?
    W_bigram = np.log(bigram_probs)
    bigram_losses = []

    def softmax(a):
        a = a - a.max()
        exp_a = np.exp(a)
        return exp_a / exp_a.sum(axis=1, keepdims=True)

    t0 = datetime.now()

    logistic.train()

    #0print(list(logistic.parameters()))

    for epoch in range(epochs):
        # shuffle sentences at each epoch
        random.shuffle(sentences)

        j = 0 # keep track of iterations
        for sentence in sentences:
            # convert sentence into one-hot encoded inputs and targets
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            print(sentence)
            #inputs = sentence[:n-1]
            #targets = sentence[1:]

            inputs = np.zeros((n - 1, V))
            targets = np.zeros((n - 1, V))
            inputs[np.arange(n - 1), sentence[:n-1]] = 1
            targets[np.arange(n - 1), sentence[1:]] = 1
            targets_bigram = targets.copy()
            inputs_bigram = inputs.copy()
            targets = np.argmax(targets, axis=1)
            targets = torch.from_numpy(targets).long().cuda()
            inputs = np.argmax(inputs, axis=1)
            #print(inputs.shape)
            inputs = torch.from_numpy(inputs).cuda().long()
            #print(inputs.shape)

            # get output predictions
            #predictions = softmax(inputs.dot(W))
            predictions = logistic(inputs)
            # do a gradient descent step
            #W = W - lr * inputs.T.dot(predictions - targets)

            # keep track of the loss
            #loss = -np.sum(targets * np.log(predictions)) / (n - 1)
            #print(predictions.shape)
            #print(targets.shape)
            
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            # keep track of the bigram loss
            # only do it for the first epoch to avoid redundancy
            if epoch == 0:
                bigram_predictions = softmax(inputs_bigram.dot(W_bigram))
                bigram_loss = -np.sum(targets_bigram * np.log(bigram_predictions)) / (n - 1)
                bigram_losses.append(bigram_loss)


            if j % 250 == 0:
                print("epoch:", epoch, "sentence: %s/%s" % (j, len(sentences)), "loss:", loss.item())
            j += 1

        torch.save(logistic.state_dict(), "models/logistic_pytorch_{}.pth".format(epoch+1))

    print("Elapsed time training:", datetime.now() - t0)
    plt.plot(losses)

    # plot a horizontal line for the bigram loss
    avg_bigram_loss = np.mean(bigram_losses)
    print("avg_bigram_loss:", avg_bigram_loss)
    plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')


    # plot smoothed losses to reduce variability
    def smoothed_loss(x, decay=0.99):
        y = np.zeros(len(x))
        last = 0
        for t in range(len(x)):
            z = decay * last + (1 - decay) * x[t]
            y[t] = z / (1 - decay ** (t + 1))
            last = z
        return y

    plt.plot(smoothed_loss(losses))
    plt.show()

    # plot W and bigram probs side-by-side
    # for the most common 200 words
    plt.subplot(1,2,1)
    plt.title("Logistic Model")
    plt.imshow(softmax(W))
    plt.subplot(1,2,2)
    plt.title("Bigram Probs")
    plt.imshow(bigram_probs)
    plt.show()




