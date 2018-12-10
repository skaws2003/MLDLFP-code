'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy as np
#from utils.graph_definition import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EfficientRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2, num_split=3, penalty=0.5, device = 'cpu', batch_first = True):
        super(EfficientRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_split = num_split
        self.rnns = []    #dim 0 is split, dim 1 is layer
        self.device = device
        self.layer_weights = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.batch_first = batch_first

        for i in range(num_split):
            l = []
            l.append(nn.GRUCell(self.input_size, self.hidden_size).to(self.device)) #first layer
            for j in range(num_layers-1):
                l.append(nn.GRUCell(self.hidden_size, self.hidden_size).to(self.device)) #other layer
            self.rnns.append(l)

        self.selective_layer = nn.Linear(self.hidden_size + self.input_size, num_split)
        #self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden=None, penalty=0.7):
        # Set initial states
        is_packed = isinstance(x, PackedSequence) #check use packed sequence
        if is_packed:
            input, batch_sizes = x
            max_batch_size = batch_sizes[0]
            input=input.view(-1, max_batch_size, self.input_size).transpose(0,1).to(self.device)
        else:
            batch_sizes = None
            max_batch_size = x.size(0) if self.batch_first else x.size(1)
            input = x

        if hidden is None:
            h0 = torch.zeros(max_batch_size, self.num_layers, self.hidden_size).to(self.device) #input hidden
            #c0 = torch.zeros(x.size(0), self.hidden_size).to(device)
        else:
            h0 = hidden


        penalty_layer = torch.ones(self.num_split, requires_grad=False).to(self.device) #add penalty layer
        cur_cell = 0


        h_c = self.rnns[cur_cell][0](input[:, 0, :], h0[:, 0, :]).unsqueeze(1) #first layer prop
        h = h_c
        for i in range(1, self.num_layers):
            h_c = self.rnns[cur_cell][i](h_c.squeeze(1), h0[:, i, :]).unsqueeze(1)
            h = torch.cat((h, h_c), dim=1)
        last_hidden = h


        penalty_layer[cur_cell] = penalty*penalty_layer[cur_cell]
        penalty_layer = penalty_layer/penalty_layer.max()

        outputs = h_c

        #first prop end

        for i in range(1, input.size(1)): #until end of sequence

            #finding which cell to use

            sum_hidden = self.layer_weights(h)
            if self.num_layers > 1:
                energy = self.layer_weights(h)
                layer_energies = torch.sum(energy, dim=2)
                sum_hidden = layer_energies.unsqueeze(1).bmm(h)


            sum_hidden = self.selective_layer(torch.cat((sum_hidden.squeeze(1), input[:,i,:]), dim=1)) #select which cell to use on next
            sum_hidden = torch.sum(sum_hidden, dim=0) #use sum of all batch


            sum_hidden = F.softmax(sum_hidden)*penalty_layer
            _, cur_cell = sum_hidden.max(0) #we use maximum mean of batch value of linear layer
            cur_cell = cur_cell.item()

            h_c = self.rnns[cur_cell][0](input[:, i, :], last_hidden[:, 0, :]).unsqueeze(1)  # continue prop
            h = h_c
            for j in range(1, self.num_layers):
                h_c = self.rnns[cur_cell][j](h_c.squeeze(1), last_hidden[:, j, :]).unsqueeze(1)
                h = torch.cat((h, h_c), dim=1)
            last_hidden = h

            penalty_layer[cur_cell] = penalty * penalty_layer[cur_cell]
            penalty_layer = penalty_layer / penalty_layer.max()

            outputs = torch.cat((outputs, h_c), dim=1)

        #outputs = outputs.transpose(0, 1)
        #out = self.fc(out)
        return outputs.transpose(0, 1), h[:,-1,:]


def test():
    batch_size = 2
    input_size = 5
    hidden_size = 4
    seq_length = 30
    num_layers = 2
    num_classes = 2
    num_split = 3
    net = EfficientRNN(input_size, hidden_size, num_layers, num_classes, num_split, device=device).to(device)
    x = torch.ones(batch_size, seq_length, input_size).to(device) # (batch, seq_length, input_size)
    y, h = net(x)
    print(y.size())

if __name__=="__main__":
    test()