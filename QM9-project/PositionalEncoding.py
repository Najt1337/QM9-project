#%%
import numpy as np
import torch
import torch.nn as nn
# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # Create a vector with positions (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term based on even and odd indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        
        # Register buffer so it's not a model parameter but still gets moved to the correct device
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
#%%
'''

d_model = 16  # Dimensionality of embeddings
max_len = 65  # Maximum sequence length
dataset_size = 10  # Number of sequences
pos_enc = PositionalEncoding(d_model, max_len)
    
input_tensor = torch.ones(dataset_size, max_len, d_model)  # Example input of shape (dataset_size, seq_len, d_model)
output = pos_enc(input_tensor)
print(output.shape)  

'''