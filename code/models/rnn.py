'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# classic GRU model
# default RNN for testing in experiment
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device = 'cpu', batch_first = True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.device = device
        self.batch_first = True
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        #self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x, hidden=None):
        # Set initial states
        is_packed = isinstance(x, PackedSequence) #check use packed sequence
        if is_packed:
            input, batch_sizes = x
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = x.size(0) if self.batch_first else input.size(1)

        if hidden is None:
            h0 = torch.zeros(self.num_layers, max_batch_size, self.hidden_size).to(self.device)  # 2 for bidirection
            #c0 = torch.zeros(x.size(0), self.hidden_size).to(device)
        else:
            h0 = hidden

        # Forward propagate LSTM
        out, h = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2) #inputshape = (batch_size, seq_len, input_size)

        return out, h.transpose(0, 1)[:,-1,:]

def test():
    input_size = 10
    hidden_size = 20
    seq_length = 5
    num_layers = 1
    num_classes = 5
    net = RNN(input_size, hidden_size, num_layers, device=device).to(device)
    x = torch.randn(1, 5, 10).to(device) # (batch, seq_length, input_size)
    y = net(x)
    print(y[0].size())

if __name__== "__main__":
    test()
