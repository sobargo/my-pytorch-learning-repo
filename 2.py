import torch
from torch.optim import SGD
import numpy
x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]
X = torch.tensor(x).float()
Y = torch.tensor(y).float()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)
import torch.nn as nn
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_later = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)
    def forward(self,x):
        x = self.input_to_hidden_later(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x
loss_func = nn.MSELoss()
mynet = MyNeuralNet().to(device)
_Y = mynet(X)
# for par in mynet.parameters():
#     print(par)
loss_value = loss_func(_Y,Y)
# print(loss_value)
opt = SGD(mynet.parameters(),lr=0.001)
loss_history = []
for _ in range(50):
    opt.zero_grad()
    loss_value = loss_func(mynet(X),Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value.cpu().detach().numpy())
    print(loss_value)
val_x = [[10,11]]
val_x = torch.tensor(val_x).float().to(device)
print(mynet(val_x))
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.title('loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()