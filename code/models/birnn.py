'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(x.size(0), self.num_layers, self.hidden_size).to(device)
        c0 = torch.zeros(x.size(0), self.num_layers, self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2) #inputshape = (batch_size, seq_len, input_size)

        out = self.fc(out)[:,-1,:]
        return out

def test():
    input_size = 10
    hidden_size = 20
    seq_length = 5
    num_layers = 1
    num_classes = 5
    net = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    x = torch.randn(1, 5, 10).to(device) # (batch, seq_length, input_size)
    y = net(x)
    print(y.size())

if __name__== "__main__":
    test()
