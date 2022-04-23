import pickle
from re import M

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ImageOps import invert
from torch.utils.data import DataLoader, Dataset, TensorDataset


class MyDS(Dataset):
    def __init__(self, X, y):
        self.samples = torch.Tensor(X)
        self.labels = torch.LongTensor(y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.labels[idx])


class ImageDataDS(MyDS):
    def __init__(self, data_split, transform=None):
        super().__init__(data_split.samples, data_split.labels)
        self.transform = transform

    def __getitem__(self, idx):
        sample, label = super().__getitem__(idx)
        sample = sample.view(1, 20, 20) / 255
        if self.transform:
            sample = self.transform(sample)
        return (sample, label)


class ImageData:
    def __init__():
        # train_ds
        # test_ds
        # images_train
        # images_test
        # names_train
        # names_test
        # dloader
        # mapping

        pass


class MetaLoader(object):
    def __init__(self, taskset, args, train=True):
        bs = args.batch_size if train else 64

        self.tasks = taskset
        self.loaders = []
        for X, y in self.tasks:
            ds = TensorDataset(X, y)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=False)
            self.loaders.append(dl)

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)
