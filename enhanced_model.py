# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


class c2p_NN_enhanced(nn.Module):
    def __init__(self):
        super(c2p_NN_enhanced, self).__init__()
        self.fc1 = nn.Linear(3,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,64)
        self.fc4 = nn.Linear(64,1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(64)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def compute_bounds(self, x):
        t, q, r = x.unbind(dim=1)
        k = r / (1 + q)
        zm = 0.5 * k / torch.sqrt(1 - (0.5 * k)**2 + 1e-8)
        zp = 1e-06 + k / torch.sqrt(1 - k**2 + 1e-8)
        return zm, zp

    def rescale(self, zm, zp, y):
        zm = zm.unsqueeze(1)  # Shape: [8192] -> [8192, 1]
        zp = zp.unsqueeze(1)  # Shape: [8192] -> [8192, 1]
        return zm + (zp-zm)*y

    def clamp(self,x):
        #y = torch.nn.functional.softplus(x)  # Smooth non-negative output
        #y = y / (1 + y)                               # Rescale to [0, 1]
        return torch.nn.functional.sigmoid(x)

    #def forward(self, x):
    #    zm, zp = self.compute_bounds(x)
    #    x = self.activation(self.bn1(self.fc1(x)))
    #    x = self.activation(self.bn2(self.fc2(x)))
    #    x = self.activation(self.bn3(self.fc3(x)))
    #    x = self.clamp(self.fc4(x))
    #    x = self.rescale(zm, zp, x)
    #    return x

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x