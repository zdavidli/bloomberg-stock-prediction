import os
import argparse
import datetime
import six
import math
import pickle

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


from models import LSTM
from models import StockLSTM
from models import LSTM2
import util 

model = StockLSTM(64)
model.cuda()
loss_function = F.mse_loss #nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-6)


data = pickle.load(open('data.dat', 'rb'))
m, n = data.shape
stocks = 2
data = np.reshape(data[:stocks], (1, stocks * n))
# data = np.reshape(data[:30], (1, 30 * n))
# Xd, yd = util.create_batches(data, batch_length=256)
Xd, yd = util.sliding_window(data, batch_length=128, overlap=64)

# train on one stock

split = 0.7
# print(Xd.shape)
X = Variable(torch.Tensor(Xd[0,:int(len(Xd[0]) * split),:])).cuda()
y = Variable(torch.Tensor(yd[0,:int(len(Xd[0]) * split),:])).cuda()
# X = Variable(torch.Tensor(Xd[0,:,:])).cuda()
# y = Variable(torch.Tensor(yd[0,:,:])).cuda()

Xtest = Variable(torch.Tensor(Xd[0,int(len(Xd[0]) * split):,:])).cuda()
ytest = Variable(torch.Tensor(yd[0,int(len(Xd[0]) * split):,:])).cuda()
control = Variable(torch.Tensor(np.reshape(Xd[0,int(len(Xd[0]) * split):,-1], (len(Xd[0]) - int(len(Xd[0]) * split), 1)))).cuda()


#print('X', X.size())
#print('y', y.size())

#sys.exit()

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
epochs = 4000
for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()
    # Step 3. Run our forward pass.
    results = model(X)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(results, y)
    loss.backward()
    optimizer.step()

    # print(epoch, loss)
    train = loss.cpu().data.numpy()
    val = 0

    with torch.no_grad():

        results = model(Xtest)
        loss = loss_function(results, ytest)
        val = loss.cpu().data.numpy()
        # print(loss)
        # results = Variable(torch.Tensor(Xtest.data[:,-1])).cuda()
        # loss = loss_function(control, ytest)

        # print(loss)
    print(str(train) + "," + str(val))

# PATH = 'model1.model'
# torch.save(model.state_dict(), PATH)

# model = LSTM(1, 100)
# model.load_state_dict(torch.load(PATH))

