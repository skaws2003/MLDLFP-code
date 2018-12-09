'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy as np
#from utils.graph_definition import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EfficientRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2, num_split=3, device = 'cpu'):
        super(EfficientRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_classes = num_classes
        self.rnns = []    #dim 0 is split, dim 1 is layer
        self.device = device
        self.layer_weights = torch.nn.Linear(self.hidden_size, self.num_layers)


        for i in range(num_split):
            l = []
            l.append(nn.GRUCell(self.input_size, self.hidden_size).to(self.device)) #first layer
            for j in range(num_layers-1):
                l.append(nn.GRUCell(self.hidden_size, self.hidden_size).to(self.device)) #other layer
            self.rnns.append(l)

        self.selective_layer = nn.Linear(hidden_size + input_size, num_split)
        #self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden=None):
        # Set initial states
        is_packed = isinstance(input, PackedSequence) #check use packed sequence
        if is_packed:
            input, batch_sizes = x
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = x.size(0) if self.batch_first else input.size(1)

        if hidden is None:
            h0 = torch.zeros(max_batch_size, self.num_layers, self.hidden_size).to(self.device) #input hidden
            #c0 = torch.zeros(x.size(0), self.hidden_size).to(device)
        else:
            h0 = hidden

        cur_cell = 0
        h = torch.zeros(max_batch_size, self.num_layers, self.hidden_size).to(self.device) #hidden initializer

        h[:, 0, :] = self.rnns[cur_cell][0](x[:, 0, :]).view(max_batch_size, 1, h.size(1)) #first layer prop
        for i in range(1, self.num_layers):
            h[:, i, :] = self.rnns[cur_cell][i](h[:, i-1, :]).view(max_batch_size, 1, h.size(1))

        outputs = h[:, -1, :].view(max_batch_size, 1, h.size(1))
        out = h

        for i in range(1, x.size(1)):


            energy = self.layer_weights(h)
            attn_energies = torch.sum(hidden * energy, dim=2)
            sum_hidden = F.softmax(attn_energies, dim=1).unsqueeze(1).bmm(h.transpose(1, 2))



            sum_hidden = self.selective_layer(torch.cat((sum_hidden, x[:,i,:].view(max_batch_size, self.input_size)), dim=1))

            sum_hidden = torch.sum(sum_hidden, dim=0) #use sum of all batch
            _, cur_cell = sum_hidden.max(1) #we use maximum mean of batch value of linear layer

            h = self.rnns[cur_cell](x[:,i,:])

            h[:, 0, :] = self.rnns[cur_cell][0](x[:, 0, :]).view(max_batch_size, 1, h.size(1))  # first layer prop
            for i in range(1, self.num_layers):
                h[:, i, :] = self.rnns[cur_cell][i](h[:, i - 1, :]).view(max_batch_size, 1, h.size(1))

            outputs = torch.cat((outputs, h[:, -1, :]), dim=1) #get last layer as output
            out = h

        #outputs = outputs.transpose(0, 1)
        #out = self.fc(out)
        return outputs, h.view(max_batch_size, self.hidden_size)


def test():
    input_size = 10
    hidden_size = 20
    seq_length = 5
    num_layers = 2
    num_classes = 5
    num_split = 3
    net = EfficientRNN(input_size, hidden_size, num_layers, num_classes, num_split, device=device).to(device)
    x = torch.randn(1, 5, 10).to(device) # (batch, seq_length, input_size)
    y = net(x)
    print(y[0].size())

if __name__=="__main__":
    test()