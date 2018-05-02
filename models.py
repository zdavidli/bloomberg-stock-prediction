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
        self.bn0 = nn.BatchNorm1d(1)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
            dropout=dropout,
            bidirectional=False,
            bias=True, )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size / 2)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size / 2, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # weight_ih_l[k] – the learnable input-hidden weights of the k-th layer, of shape (hidden_size * input_size) for k = 0. Otherwise, the shape is (hidden_size * hidden_size)
        # weight_hh_l[k] – the learnable hidden-hidden weights of the k-th layer, of shape (hidden_size * hidden_size)
        # bias_ih_l[k] – the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        # bias_hh_l[k] – the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)

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
        out = self.relu1(out)
        out = F.dropout(out, training=self.training, p=0.3)
        out = self.bn2(self.fc2(out))
        out = self.relu2(out)
        out = F.dropout(out, training=self.training, p=0.3)
        out = self.fc3(out)
        return out

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
