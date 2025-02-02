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
# from tqdm import tqdm

import numpy as np

torch.manual_seed(1)

cuda = torch.device('cuda') 

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        # self.bn0 = nn.BatchNorm1d(1)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=False,
            bias=True, )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.Sigmoid()
        self.relu2 = nn.ReLU()

        # weight_ih_l[k] – the learnable input-hidden weights of the k-th layer, of shape (hidden_size * input_size) for k = 0. Otherwise, the shape is (hidden_size * hidden_size)
        # weight_hh_l[k] – the learnable hidden-hidden weights of the k-th layer, of shape (hidden_size * hidden_size)
        # bias_ih_l[k] – the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        # bias_hh_l[k] – the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_length = x.size()[1]

        x = x.contiguous().view(seq_length, batch_size, -1)
        #print(x.size())

        # print(seq_length)

        # We need to pass the initial cell states
        # h0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        # c0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        # outputs, (ht, ct) = self.rnn(x, (h0, c0))
        out, _ = self.rnn(x)

        #print(outputs)
        out = out[-1]  # We are only interested in the final prediction
        
        #print(out)
        #out = self.bn1(self.fc1(out))

        #out = self.relu1(out)
        #out = F.dropout(out, training=self.training, p=0.3)
        #out = self.bn2(self.fc2(out))
        out = self.fc1(out)
        out = self.relu1(out)
        out = F.dropout(out, training=self.training, p=0.3)
        out = self.fc2(out)
        out = self.relu2(out)
        out = F.dropout(out, training=self.training, p=0.3)
        out = self.fc3(out)
        return out


class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super(LSTM2, self).__init__()

        self.hidden_size = hidden_size
        # self.bn0 = nn.BatchNorm1d(1)
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            bidirectional=False,
            bias=True, )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.Sigmoid()
        self.relu2 = nn.ReLU()

        # weight_ih_l[k] – the learnable input-hidden weights of the k-th layer, of shape (hidden_size * input_size) for k = 0. Otherwise, the shape is (hidden_size * hidden_size)
        # weight_hh_l[k] – the learnable hidden-hidden weights of the k-th layer, of shape (hidden_size * hidden_size)
        # bias_ih_l[k] – the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        # bias_hh_l[k] – the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_length = x.size()[1]

        x = x.contiguous().view(seq_length, batch_size, -1)
        #print(x.size())

        # print(seq_length)

        # We need to pass the initial cell states
        # h0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        # c0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        # outputs, (ht, ct) = self.rnn(x, (h0, c0))
        out, _ = self.rnn1(x)
        # out, _ = self.rnn2(out)

        #print(outputs)
        out = out[-1]  # We are only interested in the final prediction
        
        #print(out)
        #out = self.bn1(self.fc1(out))

        #out = self.relu1(out)
        #out = F.dropout(out, training=self.training, p=0.3)
        #out = self.bn2(self.fc2(out))
        out = self.fc1(out)
        out = self.relu1(out)
        out = F.dropout(out, training=self.training, p=0.3)
        out = self.fc2(out)
        out = self.relu2(out)
        out = F.dropout(out, training=self.training, p=0.3)
        out = self.fc3(out)
        return out


class StockLSTM(nn.Module):
    def __init__(self, hidden_size=100):
        super(StockLSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(1, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        
        self.hidden_size = hidden_size

    def forward(self, input, future = 0):
        h_t = Parameter(torch.zeros(input.size(0), self.hidden_size).float()).cuda()
        c_t = Parameter(torch.zeros(input.size(0), self.hidden_size).float()).cuda()
        h_t2 = Parameter(torch.zeros(input.size(0), self.hidden_size).float()).cuda()
        c_t2 = Parameter(torch.zeros(input.size(0), self.hidden_size).float()).cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        out = self.linear(h_t2)
        return out