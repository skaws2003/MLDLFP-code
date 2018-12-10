'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DARNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_split=3, device = 'cpu', batch_first = True):
        super(DARNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.device = device
        self.batch_first = True
        self.num_split = num_split

        self.weight_gru = nn.GRU(self.input_size, self.hidden_size//2, self.num_layers, batch_first=True, bidirectional=False).to(self.device) #gru that gives weight
        self.split_grus = [] #list of splited gru's
        self.weight_layer = nn.Linear(self.hidden_size//2, self.num_split).to(self.device)

        for i in range(self.num_split):
            self.split_grus.append(nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False).to(self.device)) #first layer


        #self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x, hidden=None):
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
            h0 = torch.zeros(self.num_layers, max_batch_size, self.hidden_size).to(self.device) #input hidden
            h1 = torch.zeros(self.num_layers, max_batch_size, self.hidden_size//2).to(self.device) #input hidden
            #c0 = torch.zeros(x.size(0), self.hidden_size).to(device)
        else:
            h0 = hidden

        # Forward propagate LSTM
        weight_out, h = self.weight_gru(input, h1)  # out: tensor of shape (batch_size, seq_length, hidden_size*2) #inputshape = (batch_size, seq_len, input_size)


        weight = self.weight_layer(weight_out)
        weight = F.softmax(weight, dim=2)
        out_total, h_total = self.split_grus[0](input, h0)


        mat = torch.ones(max_batch_size, 1, self.hidden_size).to(self.device)
        w = weight[:,:,0].unsqueeze(2).bmm(mat)
        out_total = (out_total*w).unsqueeze(3)


        for i in range(1, len(self.split_grus)):
            mat = torch.ones(max_batch_size, 1, self.hidden_size).to(self.device)
            w = weight[:,:,i].unsqueeze(2).bmm(mat)
            out, h = self.split_grus[i](input, h0)
            out = (out*w).unsqueeze(3)
            out_total = torch.cat((out_total, out), dim=3)

        out_total = torch.sum(out_total, 3)

        return out_total.transpose(0, 1), out_total[:,-1,:]

def test():
    input_size = 10
    hidden_size = 20
    seq_length = 5
    num_layers = 2
    num_classes = 5
    net = FKRNN(input_size, hidden_size, num_layers, device=device).to(device)
    x = torch.randn(3, 5, 10).to(device) # (batch, seq_length, input_size)
    y = net(x)
    print(y[0].size())

if __name__== "__main__":
    test()