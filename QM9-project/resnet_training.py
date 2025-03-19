#%%
import torch
from ResNet2 import ResNet
from qm9_preprocessing import res_test_dataset,res_val_dataset,res_train_dataset
from torch.utils.data import DataLoader
from qm9_preprocessing import num_features
# %%
batch_size=64

res_train_loader=DataLoader(res_train_dataset,batch_size=batch_size,shuffle=True)
res_val_loader=DataLoader(res_val_dataset,batch_size=batch_size,shuffle=False)
res_test_loader=DataLoader(res_test_dataset,batch_size=batch_size,shuffle=False)
#%%
hidden_dim=128
input_dim=num_features.shape[1]
num_blocks=2


model=ResNet(input_dim,hidden_dim,num_blocks)
#%%
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Optional: Calculate trainable parameters (default is True for requires_grad)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")
#%%
loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


def train(loader):
    model.train()
    train_loss=0
    for batch in loader: 
        inputs, targets = batch
        

        optimizer.zero_grad()
        

        pred=model(inputs)
        loss=loss_fn(pred,targets)
        loss.backward()
        optimizer.step()
        train_loss=loss
    return train_loss

def validate(loader):
    model.eval()
    val_loss=0
    for batch in loader:
        with torch.no_grad():
            inputs, targets = batch
            
            
            pred=model(inputs)
            loss=loss_fn(pred,targets)
            val_loss=loss
    return val_loss

def test(loader):
    model.eval()
    test_loss=0
    for batch in loader:
        with torch.no_grad():
            inputs, targets = batch
            
            
            pred=model(inputs)
            loss=loss_fn(pred,targets)
            test_loss=loss
    return test_loss

for epoch in range(0,500):
    train_loss=train(res_train_loader)
    val_loss=validate(res_val_loader)
    test_loss=test(res_test_loader)
    #scheduler.step(val_loss)
    with open("resnet_log.txt", "a") as file:
        if epoch % 1 == 0:
            file.write(f"Epoch {epoch}, "
                   f"Train loss: {train_loss:.5f}, "
                   f"Validation loss: {val_loss:.5f}, "
                   f"Test loss: {test_loss:.5f}\n")

     

# %%
'''
num_epochs=30
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for inputs, labels in res_train_loader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimize the model
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(res_train_loader)}")

# Testing loop
model.eval()  # Set model to evaluation mode
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():  # No need to track gradients during evaluation
    for inputs, labels in res_test_loader:
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()
        
        # Store predictions and actual values for later analysis
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(labels.tolist())

print(f"Test Loss: {test_loss/len(res_test_loader)}")
#%%
results=pd.DataFrame()
results['prediction']=predictions
results['actuals']=actuals
results.head()

# %%
'''