import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader

import matplotlib.pyplot as plt

from loader import mnist_data_module
from loader import mnist_data_module
from shutil import copyfile

data_path = r'P:\workspace\jan\fire_detection\dl\MNIST\data'
dataset = MNIST(root=data_path, download=True)

mnist_data = torchvision.datasets.MNIST(r'P:\workspace\jan\fire_detection\dl\MNIST\data', download=True)

data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=1)


data_set = datasets.MNIST(root=data_path,
                          train=True,
                          download=False,
                          transform=transforms.ToTensor()
                          )
test_set =  datasets.MNIST(root=data_path,
                          train=False,
                          download=False,
                          transform=transforms.ToTensor()
                          )

print(len(data_set), len(test_set))

plt.imshow(data_set[0][0][0,:,:], cmap='gray')

train_ds, val_ds = random_split(data_set, [50000, 10000])


batch_size = 128

# shuffle so that batches in each epoch are different, and this randomization helps generalize and speed up training
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val is only used for evaluating the model, so no need to shuffle
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
