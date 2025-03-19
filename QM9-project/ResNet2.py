#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# %%
class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim, affine=False)  # Avoid learning scale/shift
        self.norm2 = nn.BatchNorm1d(input_dim, affine=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)  

        self.proj = nn.Linear(input_dim, input_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        return self.act(x + residual)


class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super().__init__()
        
        self.res_blocks = nn.Sequential(*[ResNetBlock(input_dim, hidden_dim) for _ in range(num_blocks)])
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.norm1 = nn.BatchNorm1d(64, affine=False)
        self.fc2 = nn.Linear(64, 32)
        self.norm2 = nn.BatchNorm1d(32, affine=False)
        self.fc3 = nn.Linear(32, 1)  # No BatchNorm here!

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.res_blocks(x)

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.fc3(x)  # No activation to allow regression range

        return x