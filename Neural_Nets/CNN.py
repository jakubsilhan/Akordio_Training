import torch
from Akordio_Core.net_config import Config
import torch.nn as nn

"""Based on https://ieeexplore.ieee.org/document/7738895"""

class Model(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.feature_size = config.train.model.input
        self.output_features = config.train.model.output
        self.dropout = config.train.model.dropout
        self.device = device
        
        # dropout
        self.dropout_layer = nn.Dropout2d(p=self.dropout[0])

        # First Conv block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1))
        )

        # Second Conv block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1))
        )

        # Third Conv block
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(12, 9), stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_linear = nn.Conv2d(128, self.output_features, kernel_size=(1, 1), padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=(13,3))

    def forward(self, x):
        # [batch_size, timestep, feature_size]
        batch_size, timestep, feature_size = x.shape
        context = 7  # same as old model

        # Pad along time dimension
        x = x.permute(0, 2, 1)  # [batch, feature, time]
        x = nn.functional.pad(x, (context, context), "constant", 0)

        # Extract context windows
        x = x.unfold(dimension=2, size=2*context+1, step=1)  # [batch, feature, window, timestep]
        x = x.permute(0, 3, 1, 2)  # [batch, timestep, feature, window]
        x = x.reshape(batch_size * timestep, 1, feature_size, 2*context + 1) # [batch*timestep, 1, feature, context*2+1]

        # Continue as before
        out = self.conv_block_1(x)
        out = self.dropout_layer(out)
        out = self.conv_block_2(out)
        out = self.dropout_layer(out)
        out = self.conv_block_3(out)
        out = self.dropout_layer(out)
        out = self.conv_linear(out)
        avg = nn.AvgPool2d(kernel_size=(out.size(2), out.size(3)))
        out = avg(out)
        # out = self.avg_pool(out)
        out = out.squeeze(-1).squeeze(-1)  # [batch*timestep, num_chords]

        # Return to [batch, timestep, num_chords]
        out = out.view(batch_size, timestep, -1)
        return out