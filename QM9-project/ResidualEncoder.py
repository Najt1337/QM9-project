#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from qm9_preprocessing import num_features
#%%
class ResidualBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,input_dim)
        self.norm1=nn.LayerNorm(hidden_dim)
        self.norm2=nn.LayerNorm(input_dim)
        self.act=nn.ReLU()

    def forward(self,x):
        residual=x
        x=self.fc1(x)
        x=self.norm1(x)
        x=self.act(x)
        x=self.fc2(x)
        x=self.norm2(x)
        return x+residual
# %%
class ResidualEncoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_blocks):
        super().__init__()
        
        self.res_blocks=nn.Sequential(*[ResidualBlock(input_dim, hidden_dim) for _ in range(num_blocks)])
        
        self.fc1=nn.Linear(input_dim,128)
        self.norm1=nn.LayerNorm(128)
        
        self.fc2=nn.Linear(128,64)
        self.norm2=nn.LayerNorm(64)

        self.fc3=nn.Linear(64,32)
        self.norm3=nn.LayerNorm(32)

        self.fc4=nn.Linear(32,16)

        self.act=nn.ReLU()

    def forward(self,x):

        x=self.res_blocks(x)

        x=self.fc1(x)
        x=self.norm1(x)
        x=self.act(x)

        x=self.fc2(x)
        x=self.norm2(x)
        x=self.act(x)

        x=self.fc3(x)
        x=self.norm3(x)
        x=self.act(x)

        x=self.fc4(x)

        return x

'''
# %%
test=num_features[0:10].values
test_in=torch.tensor(test,dtype=torch.float)
hidden_dim=128
input_dim=num_features.shape[1]
num_blocks=7
resnet=ResidualEncoder(input_dim,hidden_dim,num_blocks)
test_out=resnet(test_in)
test_out.shape
'''
