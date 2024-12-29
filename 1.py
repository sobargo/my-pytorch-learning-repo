from copy import deepcopy
import numpy as np
x = np.array([[2,2]])
y = np.array([[0]])
W = [np.array([[-0.0053,0.3793],[-0.5820,-0.5204],[-0.2723,0.1896]],dtype=np.float32).T,np.array([-0.0140,0.5607,-0.0628],dtype=np.float32)
,np.array([0.1528,-0.1745,-0.1135],dtype=np.float32).T,np.array([-0.5516],dtype=np.float32)]
def feed_forward(inputs,outputs,weights):
    pre_hidden = np.dot(inputs,weights[0])+weights[1]
    hidden = 1/(1+np.exp(-pre_hidden))
    pred_out = np.dot(hidden,weights[2])+weights[3]
    mean_squared_error = np.mean(np.square(pred_out-outputs))
    return mean_squared_error
def update_weights(inputs,outputs,weights,lr):
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs,outputs,original_weights)
    for i,layer in enumerate(original_weights):
        for index,weight in np.ndenumerate(layer):
            temp_weights = deepcopy(weights)
            temp_weights[i][index] = weight + 0.0001
            _loss_plus = feed_forward(inputs,outputs,temp_weights)
            grad = (_loss_plus-original_loss)/(0.0001)
            updated_weights[i][index] -= grad*lr
    return updated_weights,original_loss
losses = []
for epoch in range(100):
    W,loss = update_weights(x,y,W,0.01)
    losses.append(loss)
print(W)
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title("loss over increasing number of epochs")
plt.xlabel("epochs")
plt.ylabel('loss value')
plt.grid(True)
plt.show()
