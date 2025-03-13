#%%
import torch
import torch.nn as nn
import numpy as np
from qm9_preprocessing import vocab_size,padding_id,SMILES_tokens_id,max_length
from PositionalEncoding import PositionalEncoding

#%%

class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,max_seq_length,padding_id,d_model,nhead,num_layers):
        super().__init__()
        self.padding_id=padding_id

        self.emb=nn.Embedding(vocab_size,d_model)
        self.pos_enc=PositionalEncoding(d_model,max_seq_length)
        
        self.encoder_layer=nn.TransformerEncoderLayer(d_model,nhead)
        self.trnsfrmr=nn.TransformerEncoder(self.encoder_layer,num_layers)
        
        
        self.act=nn.GELU()

        
        self.lin1=nn.Linear(d_model,d_model//2)
        self.lin2=nn.Linear(d_model//2,d_model//4)
        self.lin3=nn.Linear(d_model//4,d_model//8)

    def forward(self,x):
        mask=(x==self.padding_id)

        x=self.emb(x)
        x=self.pos_enc(x)

        x=x.permute(1,0,2)
        x=self.trnsfrmr(x,src_key_padding_mask=mask)
        x=x.permute(1,0,2)

        x=self.lin1(x)
        x=self.act(x)

        x=self.lin2(x)
        x=self.act(x)

        x=self.lin3(x)

        return x
    
#%%
vocab_size=vocab_size
pad_id=padding_id
max_seq=max_length
d_model=128
nhead=8
num_layers=7

encoder=TransformerEncoder(vocab_size,max_seq,pad_id,d_model,nhead,num_layers)
'''
#%%
test_in=torch.tensor(SMILES_tokens_id[0:10])
test_out=encoder(test_in)
test_out.shape
'''