from . import *
from rnn import RNN
from ernn import EfficientRNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
#from utils.graph_definition import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, net, n_layers=1, num_split=3, dropout=0, device='cpu'):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.num_split = num_split

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        if num_split != -1:
            self.model = net(input_size, hidden_size, n_layers, num_split=num_split, device=device).to(device)
        else:
            self.model = net(input_size, hidden_size, n_layers, device=device).to(device)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # Forward pass through GRU
        outputs, hidden = self.model(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        ##we don't use bidirectional here
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(2)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, net, n_layers=1, num_split=3, dropout=0.1, device='cpu'):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.model = net(input_size, hidden_size, n_layers, num_split=num_split, device=device)
        self.concat = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.model(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

class linear_decoder(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, num_split=3, dropout=0.1):
        super(linear_decoder, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(last_hidden, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)

        concat_input = torch.cat((last_hidden, context), 2)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        # Return output and final hidden state
        output = F.log_softmax(output, dim=2)
        output = output.squeeze(1)
        return output, last_hidden

def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

if __name__ == "__main__":
    embedding = nn.Embedding(10, 3, padding_idx=0)
    hidden_size = 4
    output_size = 2
    input_size = 3
    inputs = torch.LongTensor(zeroPadding([[1,2,3], [1, 2, 3, 5, 7]])).transpose(0, 1)

    encoder = EncoderRNN(input_size, hidden_size, embedding, EfficientRNN, n_layers=1, num_split=-1, dropout=0)
    #attn_model = Attn('general', hidden_size)
    #decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, output_size, EfficientRNN, n_layers=1, num_split=3, dropout=0.1)
    decoder = linear_decoder('general', embedding, hidden_size, output_size, n_layers=1, num_split=3, dropout=0.1)

    outputs, hidden = encoder(inputs, torch.LongTensor([len(seq) for seq in inputs]))
    outputs = outputs.transpose(0, 1)
    output, hidden = decoder(hidden, outputs)
    print(output)

