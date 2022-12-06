import torch
import torch.nn as nn
import numpy as np
import os
import random


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(num_features=self.in_channels)
        self.batchnorm2 = nn.BatchNorm2d(num_features=self.in_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)

        out += residual
        out = self.relu(out)

        return out


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    seed_everything()
    input_image = torch.randn(1, 1, 3, 3)

    residualblock = ResidualBlock(1)
    result = residualblock(input_image)

    eq = torch.allclose(result, torch.tensor([[[[0.1642, 0.0969, 0.0000],
                                                [0.9133, 0.0000, 0.0000],
                                                [3.6363, 0.0000, 0.9017]]]]), atol=1e-4
                        )

    print(result, eq)
