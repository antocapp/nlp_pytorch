# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime
# from util import find_analogies

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances


from glob import glob

import os
import sys
import string

sys.path.append(os.path.abspath('..'))
from rnn_class.util import get_wikipedia_data
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab as get_brown

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CUDA = torch.cuda.is_available()

class Autoencoder(nn.Module):
    def __init__(self, V, D):
        super(Autoencoder, self).__init__()
       
        self.encoder = nn.Embedding(V, D, sparse=True)
        self.decoder = nn.Embedding(V, D, sparse=True)
        self.embedding_dim = D
        self.init_emb()
    def init_emb(self):
        initrange = 0.5 / self.embedding_dim
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-0, 0)
        
    def forward(self, u_pos, u_neg, v_pos):
        #x = self.tanh(self.encoder(input_vec))
        #x = self.decoder(x)
        #return F.log_softmax(x, dim=0)
        embed_u = self.encoder(u_pos)
        embed_v = self.decoder(v_pos)

        score  = torch.mul(embed_u, embed_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()
        
        neg_embed_u = self.encoder(u_neg)
        
        neg_score = torch.mul(neg_embed_u, embed_v)
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1*neg_score).squeeze()

        loss = log_target + sum_log_sampled
        return -1*loss.sum()


# unfortunately these work different ways
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3




def get_wiki(max_files=6):
    V = 20000
    files = glob('../large_files/itwiki*.txt')
    files = files[:max_files]
    all_word_counts = {}
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        if word not in all_word_counts:
                            all_word_counts[word] = 0
                        all_word_counts[word] += 1
    print("finished counting")

    V = min(V, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx['<UNK>']

    sents = []
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    # if a word is not nearby another word, there won't be any context!
                    # and hence nothing to train!
                    sent = [word2idx[w] if w in word2idx else unk for w in s]
                    sents.append(sent)
    return sents, word2idx




def train_model(savedir):
    # get the data
    sentences, word2idx = get_wiki() #get_brown()


    # number of unique words
    vocab_size = len(word2idx)


    # config
    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negatives = 5 # number of negative samples to draw per input word
    epochs = 20
    D = 50 # word embedding size



    # learning rate decay
    learning_rate_delta = (learning_rate - final_learning_rate) / epochs


    # params
    logistic = Autoencoder(vocab_size, D)
    optimizer = torch.optim.SGD(logistic.parameters(), lr=0.1)

    # distribution for drawing negative samples
    p_neg = get_negative_sampling_distribution(sentences, vocab_size)

    if CUDA:
        logistic.to(torch.device("cuda"))

    # save the costs to plot them per iteration
    costs = []


    # number of total words in corpus
    total_words = sum(len(sentence) for sentence in sentences)
    print("total number of words in corpus:", total_words)

    # for subsampling each sentence
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)


    # train the model
    for epoch in range(epochs):
        # randomly order sentences so we don't always see
        # sentences in the same order
        np.random.shuffle(sentences)

        # accumulate the cost
        cost = 0
        counter = 0
        t0 = datetime.now()
        for sentence in sentences:
            # keep only certain words based on p_neg
            sentence = [w for w in sentence \
                if np.random.random() < (1 - p_drop[w])
            ] # remove most frequent words using p_drop
            if len(sentence) < 2:
                continue


            # randomly order words so we don't always see
            # samples in the same order
            randomly_ordered_positions = np.random.choice(
                len(sentence),
                size=len(sentence),#np.random.randint(1, len(sentence) + 1),
                replace=False,
            )

            
            for pos in randomly_ordered_positions:
                # the middle word
                word = sentence[pos]
                neg_word = np.random.choice(vocab_size, p=p_neg)

                input_pos = torch.LongTensor([word]).cuda()
                input_neg = torch.LongTensor([neg_word]).cuda()

                # get the positive context words/negative samples
                context_words = get_context(pos, sentence, window_size)
                targets = torch.LongTensor(context_words).cuda()

                # do one iteration of stochastic gradient descent
                optimizer.zero_grad()
                loss = logistic(input_pos, input_neg, targets)
                loss.backward()
   
                optimizer.step()

            counter += 1
            if counter % 100 == 0:
                print("processed %s / %s" % (counter, len(sentences)))
                print("epoch:", epoch, "sentence: %s/%s" % (epoch, len(sentences)), "loss:", loss.item())
                #sys.stdout.flush()
                # break

        # print stuff so we don't stare at a blank screen
        dt = datetime.now() - t0
        print("epoch complete:", epoch, "cost:", cost, "dt:", dt)

        # save the cost
        costs.append(cost)
        
        torch.save(logistic.state_dict(), "models/word2vec_pytorch_{}.pth".format(epoch+1))
    s = logistic.state_dict()

    # plot the cost per iteration
    plt.plot(costs)
    plt.show()

    W = s["encoder.weight"].numpy()
    V = s["decoder.weight"].numpy()

    # save the model
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)

    np.savez('%s/weights.npz' % savedir, W, V)

    # return the model
    return word2idx, W, V


def get_negative_sampling_distribution(sentences, vocab_size):
    # MODIFIED UNIGRAM DISTRIBUTION
    # Pn(w) = prob of word occuring
    # we would like to sample the negative samples
    # such that words that occur more often
    # should be sampled more often

    word_freq = np.zeros(vocab_size)
    word_count = sum(len(sentence) for sentence in sentences)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1

    # smooth it
    p_neg = word_freq**0.75

    # normalize it
    p_neg = p_neg / p_neg.sum()

    assert(np.all(p_neg > 0))
    return p_neg


def get_context(pos, sentence, window_size):
    # input:
    # a sentence of the form: x x x x c c c pos c c c x x x x
    # output:
    # the context word indices: c c c c c c

    start = max(0, pos - window_size)
    end_  = min(len(sentence), pos + window_size)

    context = []
    for ctx_pos, ctx_word_idx in enumerate(sentence[start:end_], start=start):
        if ctx_pos != pos:
            # don't include the input word itself as a target
            context.append(ctx_word_idx)
    return context


def load_model(savedir):
    with open('%s/word2idx.json' % savedir) as f:
        word2idx = json.load(f)
    npz = np.load('%s/weights.npz' % savedir)
    W = npz['arr_0']
    V = npz['arr_1']
    return word2idx, W, V



def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape

    # don't actually use pos2 in calculation, just print what's expected
    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("Sorry, %s not in word2idx" % w)
            return

    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]

    vec = p1 - n1 + n2

    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]

    # pick one that's not p1, n1, or n2
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    # print("keep_out:", keep_out)
    for i in idx:
        if i not in keep_out:
            best_idx = i
            break
    # print("best_idx:", best_idx)

    print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[best_idx], neg2))
    print("closest 10:")
    for i in idx:
        print(idx2word[i], distances[i])

    print("dist to %s:" % pos2, cos_dist(p2, vec))


def test_model(word2idx, W, V):
    # there are multiple ways to get the "final" word embedding
    # We = (W + V.T) / 2
    # We = W

    idx2word = {i:w for w, i in word2idx.items()}

    for We in (W, (W + V.T) / 2):
        print("**********")

        analogy('re', 'uomo', 'regina', 'donna', word2idx, idx2word, We)



if __name__ == '__main__':
    word2idx, W, V = train_model('w2v_model')
    # word2idx, W, V = load_model('w2v_model')
    test_model(word2idx, W, V)