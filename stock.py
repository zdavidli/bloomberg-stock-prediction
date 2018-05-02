import os
import argparse
import datetime
import six
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from tqdm import tqdm

import numpy as np

print(torch.__version__)

torch.manual_seed(1)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            dropout=dropout,
            bidirectional=False, )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]
        seq_length = x.size()[1]

        x = x.contiguous().view(seq_length, batch_size, -1)

        # We need to pass the initial cell states
        h0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        outputs, (ht, ct) = self.rnn(x, (h0, c0))

        out = outputs[-1]  # We are only interested in the final prediction
        out = self.bn1(self.fc1(out))
        out = self.relu(out)
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return out


model = LSTM(1, 100)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)
epochs = 40
for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    # model.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    # model.hidden = model.init_hidden()

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Tensors of word indices.
    # sentence_in = prepare_sequence(sentence, word_to_ix)
    # targets = prepare_sequence(tags, tag_to_ix)

    # Step 3. Run our forward pass.
    results = model(x)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(results, y)
    loss.backward()
    optimizer.step()
    # for sentence, tags in training_data:
    #     # Step 1. Remember that Pytorch accumulates gradients.
    #     # We need to clear them out before each instance
    #     model.zero_grad()

    #     # Also, we need to clear out the hidden state of the LSTM,
    #     # detaching it from its history on the last instance.
    #     model.hidden = model.init_hidden()

    #     # Step 2. Get our inputs ready for the network, that is, turn them into
    #     # Tensors of word indices.
    #     sentence_in = prepare_sequence(sentence, word_to_ix)
    #     targets = prepare_sequence(tags, tag_to_ix)

    #     # Step 3. Run our forward pass.
    #     tag_scores = model(sentence_in)

    #     # Step 4. Compute the loss, gradients, and update the parameters by
    #     #  calling optimizer.step()
    #     loss = loss_function(tag_scores, targets)
    #     loss.backward()
    #     optimizer.step()

# See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)

#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(tag_scores)

###################################################################################
###################################################################################
###################################################################################
###################################################################################

# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)


# training_data = [
#     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]
# word_to_ix = {}
# for sent, tags in training_data:
#     for word in sent:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
# tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# # These will usually be more like 32 or 64 dimensional.
# # We will keep them small, so we can see how the weights change as we train.
# EMBEDDING_DIM = 6
# HIDDEN_DIM = 6

# ######################################################################
# # Create the model:


# class LSTMTagger(nn.Module):

#     def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
#         super(LSTMTagger, self).__init__()
#         self.hidden_dim = hidden_dim

#         self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)

#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
#         self.hidden = self.init_hidden()

#     def init_hidden(self):
#         # Before we've done anything, we dont have any hidden state.
#         # Refer to the Pytorch documentation to see exactly
#         # why they have this dimensionality.
#         # The axes semantics are (num_layers, minibatch_size, hidden_dim)
#         return (torch.zeros(1, 1, self.hidden_dim),
#                 torch.zeros(1, 1, self.hidden_dim))

#     def forward(self, sentence):
#         embeds = self.word_embeddings(sentence)
#         lstm_out, self.hidden = self.lstm(
#             embeds.view(len(sentence), 1, -1), self.hidden)
#         tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_scores = F.log_softmax(tag_space, dim=1)
#         return tag_scores

######################################################################
# Train the model:


# model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM)
# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# # Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)

# for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         model.zero_grad()

#         # Also, we need to clear out the hidden state of the LSTM,
#         # detaching it from its history on the last instance.
#         model.hidden = model.init_hidden()

#         # Step 2. Get our inputs ready for the network, that is, turn them into
#         # Tensors of word indices.
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = prepare_sequence(tags, tag_to_ix)

#         # Step 3. Run our forward pass.
#         tag_scores = model(sentence_in)

#         # Step 4. Compute the loss, gradients, and update the parameters by
#         #  calling optimizer.step()
#         loss = loss_function(tag_scores, targets)
#         loss.backward()
#         optimizer.step()

# # See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)

#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(tag_scores)
