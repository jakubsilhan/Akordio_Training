import torch
import torch.nn as nn
import torch.nn.functional as F

from Akordio_Core.net_config import Config

"""Based on https://ieeexplore.ieee.org/document/9399463"""

# TODO add missing M5 block

class Model(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.feature_size = config.train.model.input
        self.output_features = config.train.model.output
        self.dropout = config.train.model.dropout
        self.device = device

        # Feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Deeper feature maps
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(self.dropout[0])
        )

        # Classifier projection
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(3,3), stride=(3,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(4,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout[1])
        )

        # Final projection
        self.fc = nn.Conv2d(16, self.output_features, kernel_size=1) # [batch, time_frames, output_size]

    def forward(self, x):
        # [batch, time_frames, feature_size]
        x = x.unsqueeze(1) # [batch, 1, time_frames, feature_size]
        x = x.transpose(2, 3) # [batch, 1, feature_size, time_frames]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc(x)

        x = torch.mean(x, dim=2)  # [batch, output_size, time_frames]
        x = x.permute(0, 2, 1)    # [batch, time_frames, output_size]
        return x