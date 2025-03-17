#%%
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from qm9_preprocessing import df
from qm9_preprocessing import vocab_size,padding_id,SMILES_tokens_id,max_length
from TransformerEncoder import TransformerEncoder
from ResidualEncoder import ResidualEncoder
from ResNet import ResNet

# %%
class HybridModel(nn.Module):
    def __init__(self,
   
    #TransformerEncoder parameters
    vocab_size,max_seq_length,padding_id,d_model,nhead,transformer_nlayers,
    
    #ResidualEncoder paramteres
    ResEnc_input_dim,ResEnc_hidden_dim,ResEnc_num_blocks,


    #ResNet parameters
    ResNet_num_blocks
    ):
        super().__init__()

        self.transformer_emb=TransformerEncoder(vocab_size,max_seq_length,pad_id,d_model,nhead,transformer_nlayers)
        
        self.res_emb=ResidualEncoder(ResEnc_input_dim,ResEnc_hidden_dim,ResEnc_num_blocks)

        self.ResNet=ResNet(32,64,ResEnc_num_blocks)

    def forward(self,x,y):

        transformer_emb=self.transformer_emb(x)
        transformer_emb=transformer_emb.mean(dim=1)

        res_emb=self.res_emb(y)

        emb=torch.cat((transformer_emb,res_emb),1)

        out=self.ResNet(emb)

        return out

'''
#%%
num_features=df.drop(['U0','SMILES'],axis=1)
num_features
transformer_input=torch.tensor(SMILES_tokens_id)
targets=torch.tensor(df['U0'].values,dtype=torch.float)
num_features=torch.tensor(num_features.values,dtype=torch.float)
#%%
test1in=transformer_input[0:10]
test2in=num_features[0:10]

vocab_size=vocab_size
pad_id=padding_id
max_seq_length=max_length
d_model=128
nhead=8
num_layers=7

hidden_dim=128
input_dim=num_features.shape[1]
num_blocks=7

model=HybridModel(vocab_size,max_seq_length,pad_id,d_model,nhead,num_layers,
input_dim,hidden_dim,num_blocks,
num_blocks)
out=model(test1in,test2in)
out.shape
#%%
# %%
test1in=transformer_input[0:10]
transformer_emb=TransformerEncoder(vocab_size,max_seq_length,pad_id,d_model,nhead,num_layers)
out1=transformer_emb(test1in)
out1=out1.mean(dim=1)
out1.shape
# %%
test2in=num_features[0:10]
res_emb=ResidualEncoder(input_dim,hidden_dim,num_blocks)
out2=res_emb(test2in)
out2.shape
# %%
emb=torch.cat((out1,out2),1)
emb.shape
'''