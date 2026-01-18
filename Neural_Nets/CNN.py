import torch
from Akordio_Core.Classes.NetConfig import Config
import torch.nn as nn

"""Based on https://ieeexplore.ieee.org/document/7738895"""

class Model(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.feature_size = config.train.model.input
        self.output_features = config.train.model.output
        self.dropout = config.train.model.dropout
        self.device = device
        self.timestep = config.data.preprocess.fragment_size
        
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

        self.latent_dim = 128
        self.conv_linear = nn.Conv2d(128, self.output_features, kernel_size=(1, 1), padding=0)

        # Output
        self.fc = nn.Linear(self.latent_dim, self.output_features)
        self.fc_root = nn.Linear(self.latent_dim, 13)
        self.fc_quality = nn.Linear(self.latent_dim, 15)

    def forward(self, x):
        # [batch_size, timestep, feature_size]
        batch_size, timestep, feature_size = x.shape
        context = 7
        window_size = 2 * context + 1

        # Pad along time dimension
        x = x.permute(0, 2, 1)  # [batch, feature, time]
        pad = nn.ConstantPad1d(context, 0)
        x = pad(x)

        # Extract context window
        result_tensors = []
        for i in range(batch_size):
            for j in range(self.timestep):
                tmp = x[i, :, j : j + window_size].unsqueeze(0) # Window slice
                
                if tmp.size(-1) < window_size: # Pad if short
                    padding_needed = window_size - tmp.size(-1)
                    tmp = nn.functional.pad(tmp, (0, padding_needed), "constant", 0)
                    
                result_tensors.append(tmp)

        # Concatenate windowed data
        x = torch.cat(result_tensors, dim=0)
        
        x = x.unsqueeze(1) # [batchsize * timestep, feature_size, context]

        # Convolutions
        out = self.conv_block_1(x)
        out = self.dropout_layer(out)
        out = self.conv_block_2(out)
        out = self.dropout_layer(out)
        out = self.conv_block_3(out)
        out = self.dropout_layer(out)
        out = self.conv_linear(out)
        avg = nn.AvgPool2d(kernel_size=(out.size(2), out.size(3)))
        out = avg(out)

        # Reshape
        out = out.squeeze(-1).squeeze(-1)  # [batch*timestep, num_chords]
        out = out.view(batch_size, timestep, -1) # [batch, timestep, num_chords]
        return out
    
    def forward_multiclass(self, x):
        # [batch_size, timestep, feature_size]
        batch_size, timestep, feature_size = x.shape
        context = 7
        window_size = 2 * context + 1

        # Pad along time dimension
        x = x.permute(0, 2, 1)  # [batch, feature, time]
        pad = nn.ConstantPad1d(context, 0)
        x = pad(x)

        # Extract context window
        result_tensors = []
        for i in range(batch_size):
            for j in range(self.timestep):
                tmp = x[i, :, j : j + window_size].unsqueeze(0) # Window slice
                
                if tmp.size(-1) < window_size: # Pad if short
                    padding_needed = window_size - tmp.size(-1)
                    tmp = nn.functional.pad(tmp, (0, padding_needed), "constant", 0)
                    
                result_tensors.append(tmp)

        # Concatenate windowed data
        x = torch.cat(result_tensors, dim=0)
        
        x = x.unsqueeze(1) # [batchsize * timestep, feature_size, context]

        # Convolutions
        out = self.conv_block_1(x)
        out = self.dropout_layer(out)
        out = self.conv_block_2(out)
        out = self.dropout_layer(out)
        out = self.conv_block_3(out)
        out = self.dropout_layer(out)
        avg = nn.AvgPool2d(kernel_size=(out.size(2), out.size(3)))
        out = avg(out)

        features = out.view(out.size(0), -1)  # [batch*timestep, num_chords]

        # Heads
        logits = self.fc(features)
        logits_root = self.fc_root(features)
        logits_quality = self.fc_quality(features)

        # Final reshape [batch, timestep, num_chords]
        logits = logits.view(batch_size, timestep, -1)
        logits_root = logits_root.view(batch_size, timestep, -1)
        logits_quality = logits_quality.view(batch_size, timestep, -1)
        return logits, logits_root, logits_quality