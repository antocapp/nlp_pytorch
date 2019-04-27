## markov.py 
contains an algorithm to found the table (V, V) of bigrams, 
probabilities of a word to come after the last word. Computed from counting and 
normalizing occurrencies.

## logistic_pytorch.py
a neural unit to model the bigram probabilities. Just the logistic regression. 
Trained with embedding layer from pytorch, in order to feed as input just the 
indeces of the words insted of 1-hot encoded vector that would increase the time
spent in training. 

## neural_pytorch.py
a neural network version of the bigram model. 
The structure of the network is an encoder-decoder. The encoder is an embedding layer, 
while the decoder is a Dense layer, not necessarily it works only with bias, you can also avoid it. 

## word2vec_pytorch.py
pytorch implementation of skipgram to retrieve valid word2vec embeddings. 
We try to minimise the NLLLoss negative log-likelihood loss of the neural 
network model.
It is implemented with negative sampling, drop probabilities and modified 
unigram models. It uses 2 embedding layer, so lookup tables, as described in
the paper since one is used for input word and the other for context words.


## glove_pytorch.py
pytorch implementation of GloVe. You can download pretrained GloVe embeddings, 
then store these embeddings into an embedding layer, where each index 
corresponds to the index of the word saved in word2idx dictionary.
Simpler than word2vec, trains easily. Once uploaded, attach to it an RNN block
with, seq2seq, Dense layer depending on the nlp task that you want to fine tune.

# from embeddings to recurrent neural networks
POS tagging and NER can be done by simply using categorical cross entropy 
in a logistic regression model. 
Or can be done to achieve better accuracy with RNNs (simple cells, LSTMs, GRUs). 
It is useful to understand the context of a word that can be both a noun or a verb. 
It can also be done with Hidden Markov Models, that learns a model of grammar!

Then, this embedding layer can be used as a first layer in the following 
framework. The key difference between a GRU and an LSTM is that a GRU has 
two gates (reset and update gates) whereas an LSTM has three gates (namely input, 
output and forget gates). The GRU unit controls the flow of information like the 
LSTM unit, but without having to use a memory unit. It just exposes the full 
hidden content without any control. GRU is relatively new, and from my 
perspective, the performance is on par with LSTM, but computationally more 
efficient (less complex structure as pointed out). So we are seeing it being 
used more and more.


## ENCODER

embedding = nn.Embedding(V, D) V = vocab_size, D = embedding size

rnn_network = nn.GRU(D, H, n_layers) 
H = hidden size
n_layers = number of stacked GRU to use

in forward, rnn_network gives the output and the hidden state, 
which is used for the next input word

## DECODER

dense_layer -> to compute a prediction on the tag or 
entities of the hidden states for each embedding

OR 

if you want a seq2seq model, like language translation, a bot answering questions, 
use another RNN.

The last output of the encoder network (sometimes called the context vector) 
is used as the initial hidden state of the decoder network. 
At every step of decoding, the decoder is given an input token and hidden state. 
The initial input token is the start-of-string <SOS> token, and the first hidden 
state is the context vector (the encoder’s last hidden state).