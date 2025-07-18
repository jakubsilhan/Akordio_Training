import torch
import torch.nn as nn
from CRF import CRF

class Crf(nn.Module):
    def __init__(self, num_chords, timestep):
        super(Crf, self).__init__()
        self.output_size = num_chords
        self.timestep = timestep
        self.Crf = CRF(self.output_size)

    def forward(self, probs, labels):
        prediction = self.Crf(probs)
        prediction = prediction.view(-1)
        labels = labels.view(-1, self.timestep)
        loss = self.Crf.loss(probs, labels)
        return prediction, loss

class CRNN(nn.Module):
    def __init__(self, feature_size, output_features, hidden_size, num_layers, bidirectional, device, dropout=(0.4,0,0)): # 12 + 1 and 6 + 1 for N chord
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_features = output_features
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        # Batchnorm and dropout
        self.batch_norm = nn.BatchNorm2d(1)
        self.dropout2 = nn.Dropout(p=dropout[2])

        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,5), padding=2)  # preserve sequence length
        self.conv2 = nn.Conv2d(1, 36, kernel_size=(1, self.feature_size))

        # Recurrent layers
        self.gru = nn.GRU(input_size=36, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        # Output
        self.fc = nn.Linear(self.hidden_size*2, self.output_features)

    def forward(self, x, device):
        # x : [batch_size * timestep * feature_size]
        x = x.unsqueeze(1) # [batch_size * num_channels=1 * timestep * feature_size]
        x = self.batch_norm(x)

        conv = self.relu(self.conv1(x))
        conv = self.relu(self.conv2(conv))
        conv = conv.squeeze(3).permute(0,2,1)

        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, conv.size(0), self.hidden_size).to(device)
        gru, h = self.gru(conv, h0)
        logits = self.fc(gru)
        return logits