import torch
import torch.nn as nn
import torch.nn.functional as F

from Akordio_Core.Classes.NetConfig import Config

"""Based on https://ieeexplore.ieee.org/document/9399463

!!! NOT FUNCTIONAL AT THE MOMENT!!!
"""

class Model(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.u_b = 16
        self.context = 7  # Context window size
        
        # Bin reducer
        self.cqt_reducer = CQT_Bin_Reducer()

        # B5 block
        self.b5_subnetwork = B5Unit(self.config, u_b=self.u_b)
        
        # M5 block
        self.m5_network = M5Unit(self.config, u_b=self.u_b)

    def forward(self, x):
        # [batch_size, timestep, feature_size]
        batch_size, timestep, feature_size = x.shape
        
        # Pad along time dimension
        x = x.permute(0, 2, 1)  # [batch, feature, time]
        x = F.pad(x, (self.context, self.context), "constant", 0)
        
        # Extract context windows
        inputs = []
        for i in range(batch_size):
            for t in range(timestep):
                window = x[i, :, t : t + 2*self.context + 1]  # [feature, context*2+1]
                inputs.append(window)
        
        inputs = torch.stack(inputs)  # [batch*timestep, feature, context*2+1]
        inputs = inputs.unsqueeze(1).unsqueeze(1)  # [batch*timestep, 1, 1, feature, context*2+1]
        
        # Process through Fifthnet
        x_reduced = self.cqt_reducer(inputs)
        pcr_time_series = self.b5_subnetwork(x_reduced)
        chord_probs = self.m5_network(pcr_time_series)
        
        # Reshape back to [batch_size, timestep, num_chords]
        chord_probs = chord_probs.view(batch_size, timestep, -1)
        
        return chord_probs


class CQT_Bin_Reducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = nn.AvgPool3d(
            kernel_size=(6, 1, 1),
            stride=(6, 1, 1)
        )

    def forward(self, x):
        return self.reduction(x)


class B5Unit(nn.Module):
    def __init__(self, config: Config, u_b=16):
        super().__init__()
        self.config = config

        # From table 4
        self.cqt_extractor = nn.Sequential(
            nn.ConstantPad3d(padding=(0, 0, 0, 0, 1, 1), value=0), 
            nn.Conv3d(1, 16, kernel_size=(3, 1, 1), padding=0),
            nn.ReLU(),
            nn.ConstantPad3d(padding=(0, 0, 0, 0, 1, 1), value=0),
            nn.Conv3d(16, 8, kernel_size=(3, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
        )

        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=(1,3,3), padding=0), 
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=(1,1,1), padding=0), 
            nn.ReLU()
        )

        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(1,3,3), padding=0),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(1,3,3), padding=0),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(1,3,1), stride=(1,1,1))
        )

        # Dropout
        self.dropout1 = nn.Dropout(self.config.train.model.dropout[0])

        # Third convolution block
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=(1, 1, 1), padding = 0), 
            nn.ReLU(),
            nn.Conv3d(8, u_b, kernel_size=(1, 3, 3), padding=0), 
            nn.ReLU(),
            nn.Conv3d(u_b, u_b, kernel_size=(1, 4, 1), padding=0) 
        )

        self.dropout2 = nn.Dropout(self.config.train.model.dropout[1])

        self.final_collapse = nn.Conv3d(u_b, u_b, kernel_size=(12, 1, 4), padding=0)

    def forward(self, x):
        x = self.cqt_extractor(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.dropout2(x)
        
        x = self.final_collapse(x)
        
        x = x.squeeze(4).squeeze(2).transpose(1, 2)
        
        return x
    

class M5Unit(nn.Module):
    def __init__(self, config: Config, u_b=16):
        super().__init__()
        self.config = config

        self.conv1_block = nn.Sequential(
            nn.ConstantPad1d(padding=(1, 1), value=0), 
            
            nn.Conv1d(u_b, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.Dropout(self.config.train.model.dropout[2]),
            
            nn.Conv1d(16, 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(8),
            nn.Dropout(self.config.train.model.dropout[3])
        )
        
        self.conv2_block = nn.Sequential(
            nn.ConstantPad1d(padding=(5, 6), value=0),
            
            nn.Conv1d(8, 16, kernel_size=12, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.Dropout(self.config.train.model.dropout[4]),
            
            nn.Conv1d(16, 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(8)
        )

        self.conv3_block = nn.Sequential(
            nn.ConstantPad1d(padding=(5, 6), value=0), 
            
            nn.Conv1d(8, 32, kernel_size=12, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.Dropout(self.config.train.model.dropout[5])
        )
        
        self.final_conv = nn.Conv1d(32, self.config.train.model.output, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2) 
        
        x = self.conv1_block(x) 
        
        x = self.conv2_block(x)
        
        x = self.conv3_block(x)
        
        x = self.final_conv(x)
        
        x = x.mean(dim=2, keepdim=False)
        
        return x