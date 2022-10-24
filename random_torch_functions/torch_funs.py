import torch
import torch.nn as nn
import torchvision
import numpy as np



mnist_data = torchvision.datasets.MNIST(r'P:\workspace\jan\fire_detection\dl\MNIST\data', download=True)

data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=1)


data_loader

## check first model layer
m = learn.model[0]

## see weights of layers
m[0].weight.shape

## fast ai data loader form MNIST Numbers
def get_dls(bs=64): return DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter('training','testing'),
           get_y=parent_label,
          batch_tfms=Normalize() ).dataloaders(path, bs=bs)
dls = get_dls()
dls.show_batch(max_n=9, figsize=(4,4))

## rewrite that into Pytorch Ligthning


## basic conv block
def conv(ni, nf, ks=3, act=True):
    res = nn.Conv2d(ni, nf, stride=2, kernel_size=ks, padding=ks//2)
    if act: res = nn.Sequential(res, nn.ReLU())
    return res