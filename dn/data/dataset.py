import pickle

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


class MarketCLTasks:
    def __init__(self, df):
        days = df["day"].unique()
        ndays = len(list(days))
        tr_n = int(ndays * 3 / 4)
        self.tr_n = tr_n
        self.dfTr = df.loc[df["day"] < tr_n]
        self.dfTe = df.loc[df["day"] >= tr_n]
        self.train_tasks = []
        self.test_tasks = []

    def clear_data(self):
        self.dfTr = []
        self.dfTe = []

    def load_data(self, df):
        self.dfTr = df.loc[df["day"] < self.tr_n]
        self.dfTe = df.loc[df["day"] >= self.tr_n]

    # def create_tasks(self,df,taskL):
    #     symbols=df['sym'].unique()
    #     days=df['day'].unique()
    #     for s in symbols:
    #         for d in days:
    #             dfsd=df.loc[(df['day']==d)&(df['sym']==s)]
    #             n=dfsd.shape[0]
    #             tasks=[(d,s,i) for i in range(n)]
    #             taskL+=tasks
    def get_num_train_tasks(self):
        return len(self.train_tasks)

    def get_num_test_tasks(self):
        return len(self.test_tasks)

    def get_task(self, k, kind="train"):
        if kind == "train":
            (d, s, i) = self.train_tasks[k]
            df = self.dfTr.loc[(self.dfTr["day"] == d) & (self.dfTr["sym"] == s)]
        elif kind == "test":
            (d, s, i) = self.test_tasks[k]
            df = self.dfTe.loc[(self.dfTr["day"] == d) & (self.dfTr["sym"] == s)]
        dftrain = df.iloc[0:i][
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Dividends",
                "Open_prev",
                "High_prev",
                "Low_prev",
                "Close_prev",
                "Volume_prev",
                "Dividends_prev",
                "hurst",
                "SMA_10",
                "SMA_20",
                "SMA_50",
                "SMA_200",
                "VOL_SMA_20",
                "RSI_14",
                "BBL_5_2.0",
                "BBM_5_2.0",
                "BBU_5_2.0",
                "BBB_5_2.0",
                "BBP_5_2.0",
                "MACD_12_26_9",
                "MACDh_12_26_9",
                "MACDs_12_26_9",
                "sym",
            ]
        ]
        label = df.iloc[-1][
            ["(0.02, 0.01)", "(0.01, 0.005)", "(0.01, 0.02)", "(0.005, 0.01)"]
        ]
        return dftrain, label


class MetaLoader(object):
    def __init__(self, taskset, args, train=True):
        bs = args.batch_size if train else 64

        self.tasks = taskset
        self.loaders = []
        for (X, X_), y in self.tasks:
            ds = TensorDataset(X, X_, y)
            dl = DataLoader(ds, batch_size=bs, shuffle=True, pin_memory=False)
            self.loaders.append(dl)

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)
