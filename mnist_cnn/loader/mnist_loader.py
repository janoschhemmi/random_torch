
import torch
import torch.nn as nn
import pytorch_lightning as pl


class mnist_data_module(pl.LightningDataModule):

    def __init__(self, train_set, val_set, test_set, batch_size, shuffle):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.train_dataset = Dataset(self.train_set)
        self.val_dataset   = Dataset(self.val_set)
        self.test_dataset  = Dataset(self.test_set)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            #num_workers=cpu_count()
            num_workers = 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )



