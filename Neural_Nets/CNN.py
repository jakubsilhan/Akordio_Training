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
        
        # First Conv block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1))
        )

        # Second Conv block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1))
        )

        # Third Conv block
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(12, 9), padding=0),
            nn.ReLU(inplace=True)
        )

        self.conv_linear = nn.Conv2d(128, self.output_features, kernel_size=(1, 1), padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=(13,3))

    def forward(self, x):
        # # [batch_size, timestep,feature_size]
        # x = x.unsqueeze(1) # [batch_size, 1, timestep, feature_size]
        # x = x.transpose(2, 3) # [batch, 1, feature_size, time_frames]
        # # TODO consider slicing into context windows
        # x = self.conv_block_1(x)
        # x = self.conv_block_2(x)
        # x = self.conv_block_3(x)
        # x = self.conv_linear(x)
        # x = self.avg_pool(x)
        # x = x.squeeze(-1) # [batch_size, feature_size, timestep]
        # x = x.transpose(1, 2) # [batch_size, timestep, feature_size]

        # x: [batch_size, timestep, feature_size]
        batch_size, timestep, feature_size = x.shape
        context = 7  # same as old model

        # Pad along time dimension
        x = x.permute(0, 2, 1)  # [batch, feature, time]
        x = nn.functional.pad(x, (context, context), "constant", 0)

        # Extract context windows
        inputs = []
        for i in range(batch_size):
            for t in range(timestep):
                window = x[i, :, t : t + 2*context + 1]  # [feature, context*2+1]
                inputs.append(window)
        inputs = torch.stack(inputs)  # [batch*timestep, feature, context*2+1]
        inputs = inputs.unsqueeze(1)  # [batch*timestep, 1, feature, context*2+1]

        # Continue as before
        out = self.conv_block_1(inputs)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_linear(out)
        out = self.avg_pool(out)
        out = out.squeeze(-1).squeeze(-1)  # [batch*timestep, num_chords]

        # Return to [batch, timestep, num_chords]
        out = out.view(batch_size, timestep, -1)
        return out

        # return x