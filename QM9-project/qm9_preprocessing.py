#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tokenizer import SMILESTokenizer
from itertools import repeat
import torch
from torch.utils.data import TensorDataset, Subset
from sklearn.model_selection import train_test_split
#%%
data=np.load('qm9_data.npz',allow_pickle=True)
data_smiles=pd.read_csv('QM9_SMILES.csv')
data.files
# %%
df=pd.DataFrame()
features=data.files[4::]

for feature in features:
    df[feature]=data[feature]

#%%
data_smiles.drop('Unnamed: 0',axis=1,inplace=True)
# %%
df['SMILES']=data_smiles
df.drop('gap',axis=1,inplace=True)
# %%
num_features=df.drop(['SMILES','U0'],axis=1)
targets=df['U0']
num_features
# %%
scaler = StandardScaler()
for column in num_features.columns:
    scaler.fit(num_features[column].values.reshape(-1, 1))
    num_features[column] = scaler.transform(num_features[column].values.reshape(-1, 1))

scaler2 = MinMaxScaler(feature_range=(-1, 1))
for column in num_features.columns:
    scaler2.fit(num_features[column].values.reshape(-1, 1))
    num_features[column] = scaler2.transform(num_features[column].values.reshape(-1, 1))
# %%
tokenizer=SMILESTokenizer()
SMILES_tokens_id=[]
for SMILES in df['SMILES']:
    ids=tokenizer.encode(SMILES)
    SMILES_tokens_id.append(ids)

# %%
max_length=len(max(SMILES_tokens_id,key=len))
print('SMILES max length:',max_length)
# %%
vocab_size=len(tokenizer.vocabulary)
padding_id=tokenizer.vocabulary.index('<PAD>')
print('Padding token id:',padding_id,'\n','Vocabulary size:',vocab_size)
# %%
SMILES_padded=SMILES_tokens_id
for SMILES in SMILES_padded:
    SMILES.extend(repeat(padding_id,max_length-len(SMILES)))

# %%
SMILESid_dataset=TensorDataset(torch.tensor(SMILES_tokens_id),torch.tensor(targets.values))
num_features_dataset=TensorDataset(torch.tensor(num_features.values),torch.tensor(targets.values))
idx=list(range(len(SMILESid_dataset)))
# %%
train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
# %%
transformer_train_dataset = Subset(SMILESid_dataset, train_idx)
transformer_val_dataset = Subset(SMILESid_dataset, val_idx)
transformer_test_dataset = Subset(SMILESid_dataset, test_idx)

res_train_dataset = Subset(num_features_dataset, train_idx)
res_val_dataset = Subset(num_features_dataset, val_idx)
res_test_dataset = Subset(num_features_dataset, test_idx)

