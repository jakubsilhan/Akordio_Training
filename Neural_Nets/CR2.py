import torch
from Akordio_Core.Classes.NetConfig import Config
import torch.nn as nn

"""Based on https://brianmcfee.net/papers/ismir2017_chord.pdf"""

class Model(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.feature_size = config.train.model.input
        self.hidden_size = config.train.model.hidden[0]
        self.output_features = config.train.model.output
        self.num_layers = config.train.model.layers
        self.bidirectional = config.train.model.bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.dropout = config.train.model.dropout
        self.device = device
                
        # Batchnorm
        self.batch_norm = nn.BatchNorm2d(1)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 1, (5,5), padding=2)
        self.conv2 = nn.Conv2d(1, 36, (1, self.feature_size))

        # Recurrent layers
        self.gru = nn.GRU(input_size=36, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.decoder_gru = nn.GRU(input_size=self.hidden_size*self.num_directions, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        # Output heads
        self.fc = nn.Linear(self.hidden_size*self.num_directions, self.output_features)
        self.fc_root = nn.Linear(self.hidden_size * self.num_directions, 13)
        self.fc_quality = nn.Linear(self.hidden_size * self.num_directions, 15)

    def _shared_forward(self, x):
        """Shared feature extraction for both forward methods"""
        # [batch_size, timestep, feature_size]
        # Preparations
        x = x.unsqueeze(1)  # [batch_size, num_channels=1, timestep, feature_size]
        x = self.batch_norm(x)

        # First conv
        x = self.conv1(x)
        x = self.relu(x)

        # Second conv
        x = self.conv2(x)  # [batch, num_channels=out_feature_maps, timestep, feature_size]
        x = self.relu(x)

        # Reshape
        x = x.squeeze(3) # [batch, num_channels=out_feature_maps, timestep]
        x = x.permute(0, 2, 1)  # [batch, timestep, num_channels=out_feature_maps]
        
        # Gru
        h_init = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.gru(x, h_init)
        x, _ = self.decoder_gru(x)
        return x

    def forward(self, x):
        gru = self._shared_forward(x)
        logits = self.fc(gru)
        return logits

    def forward_multitask(self, x):
        gru = self._shared_forward(x)
        logits = self.fc(gru)
        root_logits = self.fc_root(gru)
        quality_logits = self.fc_quality(gru)
        return logits, root_logits, quality_logits