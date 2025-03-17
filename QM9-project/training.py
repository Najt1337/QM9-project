#%%
import torch
from HybridModel import HybridModel
from qm9_preprocessing import transformer_test_dataset,transformer_val_dataset,transformer_train_dataset
from qm9_preprocessing import res_test_dataset,res_val_dataset,res_train_dataset
from torch.utils.data import DataLoader
from qm9_preprocessing import vocab_size,padding_id,SMILES_tokens_id,max_length,num_features
# %%
batch_size=2048

transformer_train_loader=DataLoader(transformer_train_dataset,batch_size=batch_size,shuffle=False)
transformer_val_loader=DataLoader(transformer_val_dataset,batch_size=batch_size,shuffle=False)
transformer_test_loader=DataLoader(transformer_test_dataset,batch_size=batch_size,shuffle=False)

res_train_loader=DataLoader(res_train_dataset,batch_size=batch_size,shuffle=False)
res_val_loader=DataLoader(res_val_dataset,batch_size=batch_size,shuffle=False)
res_test_loader=DataLoader(res_test_dataset,batch_size=batch_size,shuffle=False)
#%%
vocab_size=vocab_size
pad_id=padding_id
max_seq_length=max_length
d_model=128
nhead=8
num_layers=1

hidden_dim=128
input_dim=num_features.shape[1]
num_blocks=1


model=HybridModel(vocab_size,max_seq_length,pad_id,d_model,nhead,num_layers,
input_dim,hidden_dim,num_blocks,
num_blocks)
#%%
loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# %%
def train(loader1,loader2):
    model.train()
    train_loss=0
    for (batch1, batch2) in zip(loader1, loader2): 
        inputs1, targets1 = batch1
        inputs2, targets2 = batch2

        optimizer.zero_grad()
        

        pred=model(inputs1.long(),inputs2)
        loss=loss_fn(pred,targets1)
        loss.backward()
        optimizer.step()
        train_loss=loss
    return train_loss

def validate(loader1,loader2):
    model.eval()
    val_loss=0
    for (batch1, batch2) in zip(loader1, loader2):
        with torch.no_grad():
            inputs1, targets1 = batch1
            inputs2, targets2 = batch2
            
            pred=model(inputs1.long(),inputs2)
            loss=loss_fn(pred,targets1)
            val_loss=loss
    return val_loss

def test(loader1,loader2):
    model.eval()
    test_loss=0
    for (batch1, batch2) in zip(loader1, loader2):
        with torch.no_grad():
            inputs1, targets1 = batch1
            inputs2, targets2 = batch2
            
            pred=model(inputs1.long(),inputs2)
            loss=loss_fn(pred,targets1)
            test_loss=loss
    return test_loss

# %%

for epoch in range(0,500):
    train_loss=train(transformer_train_loader,res_train_loader)
    val_loss=validate(transformer_val_loader,res_val_loader)
    test_loss=test(transformer_test_loader,res_test_loader)
    if epoch%1==0:
        print(f"Epoch {epoch}, \
        Train loss: {train_loss:.5f},\
        Validation loss:{val_loss:.5f},\
        Test loss: {test_loss:.5f}")
