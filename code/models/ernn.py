'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.rnns = []
        self.device = device
        for i in range(num_split):
            self.rnns.append(nn.GRUCell(self.input_size, self.hidden_size).to(self.device))

        self.selective_layer = nn.Linear(hidden_size + input_size, num_split)  # 2 for bidirection
        #self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, hidden=None):
        # Set initial states
        if hidden is None:
            h0 = torch.zeros(1, 1, self.hidden_size).to(self.device)  # 2 for bidirection
            #c0 = torch.zeros(x.size(0), self.hidden_size).to(device)
        else:
            h0 = hidden

        cur_cell = 0

        h = self.rnns[cur_cell](x[:, 0, :])
        outputs = h.view(x.size(0), 1, h.size(1))
        out = h

        for i in range(1, x.size(1)):
            output = self.selective_layer(torch.cat((h, x[:,i,:].view(x.size(0),x.size(2))), dim=1))

            output = F.softmax(output, dim=1)
            _, cur_cell = output.max(1)

            h = self.rnns[cur_cell](x[:,i,:])
            outputs = torch.cat((outputs, h.view(x.size(0), 1, h.size(1))), dim=1)
            out = h

        #outputs = outputs.transpose(0, 1)
        #out = self.fc(out)
        return outputs.view(1, -1, self.hidden_size), h.view(1, 1, -1)


def test():
    input_size = 10
    hidden_size = 20
    seq_length = 5
    num_layers = 2
    num_classes = 5
    num_split = 3
    net = EfficientRNN(input_size, hidden_size, num_layers, num_classes, num_split, device=device)
    x = torch.randn(1, 5, 10).to(device) # (batch, seq_length, input_size)
    y = net(x)
    print(y[0].size())

if __name__=="__main__":
    test()