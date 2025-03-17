#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
# %%
class ResNetBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,2*hidden_dim)
        self.fc3=nn.Linear(2*hidden_dim,hidden_dim)
        self.fc4=nn.Linear(hidden_dim,input_dim)
        self.norm1=nn.LayerNorm(hidden_dim)
        self.norm2=nn.LayerNorm(2*hidden_dim)
        self.norm4=nn.LayerNorm(input_dim)
        self.act=nn.ReLU()
        self.drop=nn.Dropout(p=0.2)

    def forward(self,x):

        residual=x

        x=self.fc1(x)
        x=self.norm1(x)
        x=self.drop(x)
        x=self.act(x)

        x=self.fc2(x)
        x=self.norm2(x)
        x=self.drop(x)
        x=self.act(x)

        x=self.fc3(x)
        x=self.norm1(x)
        x=self.drop(x)
        x=self.act(x)

        x=self.fc4(x)
        x=self.norm4(x)

        return x+residual
    
# %%
class ResNet(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_blocks):
        super().__init__()
        
        self.res_blocks=nn.Sequential(*[ResNetBlock(input_dim, hidden_dim) for _ in range(num_blocks)])
        
        self.fc1=nn.Linear(input_dim,64)
        self.norm1=nn.LayerNorm(64)
        
        self.fc2=nn.Linear(64,32)
        self.norm2=nn.LayerNorm(32)

        self.fc3=nn.Linear(32,1)

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

        return x