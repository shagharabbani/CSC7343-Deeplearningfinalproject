import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_size = 1441
        self.hidden_size = 100
        self.output_size = 1441
        self.n_layers = 1
        self.n_epochs=10
        self.n_iters=1

        self.c1 = nn.Conv1d(self.input_size, self.hidden_size, 2)
        self.p1 = nn.AvgPool1d(2)
        self.c2 = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        self.p2 = nn.AvgPool1d(2)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=0.01)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):
        print(inputs)
        batch_size = inputs.size(0)
        hidden=None
        
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        # inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = p.transpose(1, 2).transpose(0, 1)
        
        p = F.tanh(p)
        
        output, hidden = self.gru(p, hidden)
        conv_seq_len = output.size(0)
        output = output.view(conv_seq_len * batch_size, self.hidden_size) # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = F.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden

    def train(self,x_train,y_train):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        losses = np.zeros(self.n_epochs) # For plotting

        for epoch in range(self.n_epochs):

            for iter in range(self.n_iters):
                # _inputs = sample(50)
                # inputs = Variable(torch.from_numpy(_inputs[:-1]).float())
                # targets = Variable(torch.from_numpy(_inputs[1:]).float())

                if torch.is_tensor(x_train)==False:
                    x_train = torch.from_numpy(x_train).float()
                    x_train = torch.unsqueeze(x_train,0)
                    inputs=x_train
                    # x_train = torch.unsqueeze(x_train,0)
                if torch.is_tensor(y_train)==False:
                    y_train = torch.from_numpy(y_train).float()
                    y_train = torch.unsqueeze(y_train,0)
                    targets=y_train

                # Use teacher forcing 50% of the time
                force = random.random() < 0.5
                outputs, hidden = self(inputs)

                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # losses[epoch] += loss.data[0]

            # if epoch > 0:
            #     print(epoch, loss.data[0])


    def predict(self,x):
        return self(x)

# input_size = 20
# hidden_size = 50
# output_size = 7
# batch_size = 5
# n_layers = 2
# seq_len = 15

# rnn = RNN(input_size, hidden_size, output_size, n_layers=n_layers)

# inputs = Variable(torch.rand(seq_len, batch_size, input_size)) # seq_len x batch_size x 
# outputs, hidden = rnn(inputs, None)
# print('outputs', outputs.size()) # conv_seq_len x batch_size x output_size
# print('hidden', hidden.size()) # n_layers x batch_size x hidden_size