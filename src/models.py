import torch.nn as nn
import torch

class BabyCNN(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        input_channels = config['NbFeatures']
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 30, 4, stride = 1, padding = 'same'),
            nn.ReLU(),
            nn.Conv2d(30, 15, 4, stride = 1, padding = 'same'),
            nn.ReLU(),
            ## RELU
            nn.Conv2d(15, 5, 4, stride = 1, padding = 'same'),
            nn.ReLU(),
            ## RELU
            nn.Conv2d(5, 1, 4, stride = 1, padding = 'same'),
        )

    def forward(self, x):
        return self.net(x)