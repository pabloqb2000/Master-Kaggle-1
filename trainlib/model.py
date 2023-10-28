import torch
import numpy as np
from torch import nn
from math import sqrt
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from trainlib.layers import PositionalEncoding

class TransformerModel(nn.Module):
    """
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, d_output, dropout=0.5, use_cuda=True):
        super().__init__()
        self.model_type = 'Transformer'
        self.n_token = ntoken
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.linear = nn.Linear(d_model, d_output)
        
        self.init_weights()
        
        if not torch.cuda.is_available() and use_cuda:
            print("Cuda unavailable")

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.to(self.device)

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def get_non_zero_indexes_n_weights(self, src, batch_size):
        indexes = torch.zeros((batch_size, self.d_model), dtype=torch.int32).to(self.device)
        weights = torch.zeros((batch_size, self.d_model), dtype=torch.float32).to(self.device)
        for i in range(batch_size):
            non_zero = src[i].nonzero().squeeze()
            indexes[i, :len(non_zero)] = non_zero
            indexes[i,  len(non_zero)] = 2001 # <EOS> token
            weights[i, :len(non_zero)] = src[i, non_zero]
            weights[i,  len(non_zero)] = 1.
        weights = weights.unsqueeze(-1)
        return indexes, weights

    def forward(self, src, src_mask = None):
        batch_size = src.shape[0]
        indexes, weights = self.get_non_zero_indexes_n_weights(src, batch_size)
        src = weights * self.embedding(indexes) * sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = torch.mean(output, axis=1)
        output = self.linear(output)
        return output
    
class DNN(nn.Module):
    def __init__(self, dims, dropout):
        super().__init__()

        self.layers = nn.ModuleList()
        for k in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[k], dims[k+1]))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for layer in self.layers[:-1]:
          x = layer(x)
          x = self.dropout(x)
          x = self.relu(x)
        x = self.layers[-1](x)
        # x = self.logsoftmax(x)
        return x