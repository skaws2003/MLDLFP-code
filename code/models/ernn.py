'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_definition import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EfficientRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_split):
        super(EfficientRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.rnns = []
        for i in range(num_split):
            self.rnns.append(nn.moudles.rnn.LSTMCell(self.input_size, self.hidden_size))

        self.selective_layer = nn.Linear(hidden_size * 2 + input_size, num_split)  # 2 for bidirection
        self.l1 = nn.Linear(hidden_size * 4 * 32, hidden_size * 4)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        cur_cell = 0
        outputs = []

        h, c = self.rnns[cur_cell](x[:, 0, :].view(x.size(0), x.size(2)), (h0, c0))
        outputs.append(h)

        for i in range(1, x.size(1)):
            output = self.selective_layer(torch.stack((h, x[:,i,:].view(x.size(0),x.size(2))), dim=1))

            output = F.softmax(output, dim=1)
            _, cur_cell = outputs.max(1)

            h, c = self.rnns[cur_cell](x[:,i,:].view(x.size(0), x.size(2)), (h, c))
            outputs.append(h)

        outputs = outputs.transpose(0, 1)
        return outputs #(batch, seq_len, hidden)


def test():
    input_size = 10
    hidden_size = 20
    seq_length = 15
    num_layers = 2
    num_classes = 5
    num_split = 3
    net = EfficientRNN(input_size, hidden_size, num_layers, num_classes, num_split)
    x = torch.randn(1, 15, 10) # (batch, seq_length, input_size)
    y = net(x)
    print(y)
