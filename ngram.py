import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
torch.manual_seed(1)
from time import time
# Distributional Hypothesis
# words that appear in similar contexts are related
# to each other semantically

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Meriggiare pallido e assorto - Montale
test_sentence = """Meriggiare pallido e assorto
presso un rovente muro d’orto,
ascoltare tra i pruni e gli sterpi
schiocchi di merli, frusci di serpi.
Nelle crepe del suolo o su la veccia
spiar le file di rosse formiche
che ora si rompono ed ora si intrecciano
a sommo di minuscole biche.
Osservare tra frondi il palpitare
lontano di scaglie di mare
mentre si levano tremuli scricchi
di cicale dai calvi picchi.
E andando nel sole che abbaglia
sentire con triste meraviglia
come è tutta la vita e il suo travaglio
in questo seguitare una muraglia
che ha in cima cocci aguzzi di bottiglia.""".split()

test_sentence = [t.lower().translate(str.maketrans('','',string.punctuation)) for t in test_sentence]
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
#print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)

class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss() #negative log-likelihood
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

try:
    start = time()
    for epoch in range(10000):
        total_loss = 0
        for context, target in trigrams:
            #print(context, target)
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            #print(context_idxs)
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()
            #1/0
            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)


            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
        if not epoch % 500:
            e = epoch+1
            t = time()-start
            start = time()
            printing = {"e":e, "t":t, "total_loss":total_loss}
            print("Epoch: {e} | Loss {total_loss} | elapsed time: {t}".format(**printing))
    #print(losses)  # The loss decreased every iteration over the training data!

    torch.save(model.state_dict(), "ngram.pth")

except KeyboardInterrupt:
    torch.save(model.state_dict(), "ngram.pth")
    print("Saved model at total loss: {}".format(total_loss))