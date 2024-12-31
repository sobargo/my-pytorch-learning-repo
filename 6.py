from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np 
data_folder = '~/data/FMNIST'
fmnist = datasets.FashionMNIST(data_folder,download=True,train=True)
tr_image = fmnist.data
tr_targets = fmnist.targets
class FMNISTDataset(Dataset):
    def __init__(self,x,y):
        x = x.float()/255
        x = x.view(-1,28*28)
        self.x,self.y = x,y
    def __getitem__(self, index):
        x,y = self.x[index],self.y[index]
        return x.to(device),y.to(device)
    def __len__(self):
        return len(self.x)
def get_data():
    train = FMNISTDataset(tr_image,tr_targets)
    trn_dl = DataLoader(train,batch_size=32,shuffle=True)
    return trn_dl
from torch.optim import SGD
def get_model():
    model = nn.Sequential(
        nn.Linear(28*28,1000),
        nn.ReLU(),
        nn.Linear(1000,10)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),lr=0.01)
    return model, loss_fn,optimizer
def train_batch(x,y,model,opt,loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction,y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()
@torch.no_grad()
def accuracy(x,y,model):
    model.eval()
    predition = model(x)
    max_value,argmaxes = predition.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()
trn_dl = get_data()
model,loss_fn,optimizer = get_model()
losses,accuracies = [],[]
for epoch in range(5):
    print(epoch)
    epoch_losses,epoch_accuracies = [],[]
    for ix,batch in enumerate(iter(trn_dl)):
        x,y = batch
        batch_loss = train_batch(x,y,model,optimizer,loss_fn)
        epoch_losses.append(batch_loss)
    epoch_loss = np.array(epoch_losses).mean()
    for ix,batch in enumerate(iter(trn_dl)):
        x,y = batch
        is_correct = accuracy(x,y,model)
        epoch_accuracies.extend(is_correct)
    epoch_accuracy = np.mean(epoch_accuracies)
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
print(losses)
print(accuracies)