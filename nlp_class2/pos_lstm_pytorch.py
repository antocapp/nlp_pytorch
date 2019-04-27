# LSTM expects 3D tensor input
# 1-st axis is the sequence
# 2-nd axis is the minibatch
# 3-rd axis has indexes of the input

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

def simple_example_lstm():
    lstm = nn.LSTM(3, 3)
    inputs = [torch.rand(1, 3) for _ in range(5)]

    hidden = (torch.rand(1,1,3), torch.rand(1,1,3))
    for i in inputs:
        print(i.shape)
        out, hidden = lstm(i.view(1, 1, -1), hidden)
        print(out_hidden.shape)
        print("")

    # alternatively, we can do the entire sequence all at once.
    # the first value returned by LSTM is all of the hidden states throughout
    # the sequence. the second is just the most recent hidden state
    # (compare the last slice of "out" with "hidden" below, they are the same)
    # The reason for this is that:
    # "out" will give you access to all hidden states in the sequence
    # "hidden" will allow you to continue the sequence and backpropagate,
    # by passing it as an argument  to the lstm at a later time
    # Add the extra 2nd dimension
    inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
    out, hidden = lstm(inputs, hidden)
    print(out)
    print(hidden)


