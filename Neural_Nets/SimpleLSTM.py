import torch
import torch.nn as nn
import torch.nn.functional as F

from Akordio_Core.Classes.NetConfig import Config

"""Created as a baseline model"""

class Model(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.feature_size = config.train.model.input
        self.hidden_size = config.train.model.hidden[0]
        self.output_size = config.train.model.output
        self.num_layers = config.train.model.layers
        self.dropout = config.train.model.dropout[0]
        self.bidirectional = config.train.model.bidirectional
        self.device = device
        
        # LSTM layers
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, self.num_layers, dropout=self.dropout, batch_first=True, bidirectional=self.bidirectional)
        # Output layer
        if self.bidirectional:
            hidden_n = self.hidden_size*2
        else:
            hidden_n = self.hidden_size
        self.output_layer = nn.Linear(hidden_n, self.output_size)

    def forward(self, X):
        if self.lstm.bidirectional:
            hidden_n = self.num_layers*2
        else:
            hidden_n = self.num_layers
        hidden_states = torch.zeros(hidden_n, X.size(0), self.hidden_size).to(self.device)
        cell_states = torch.zeros(hidden_n, X.size(0), self.hidden_size).to(self.device)

        # Forward pass through the LSTM
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out)        
        return out
