from torchvision import datasets
import torch
data_folder = '~/data/FMNIST'
fmnist = datasets.FashionMNIST(data_folder, download=True,train=True)
tr_images= fmnist.data
tr_targets = fmnist.targets
unique_values = tr_targets.unique()

import matplotlib.pyplot as plt
import numpy as np
R,C= len(tr_targets.unique()),10
fig,ax = plt.subplots(R,C,figsize=(10,10))
for label_class,plot_row in enumerate(ax):
    label_x_rows = np.where(tr_targets == label_class)[0]
    for plot_cell in plot_row:
        plot_cell.grid(False);plot_cell.axis('off')
        ix = np.random.choice(label_x_rows)
        x,y = tr_images[ix],tr_targets[ix]
        plot_cell.imshow(x,cmap='gray')
plt.tight_layout()
plt.show()