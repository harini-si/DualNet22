import argparse
import pickle

import learn2learn as l2l
import torch
from learn2learn.data import TaskDataset
from torch.utils.data import DataLoader

from dn.data import (
    BarlowAugment,
    ContinousNWays,
    ImageData,
    ImageDataDS,
    MetaLoader,
    MyDS,
)
from dn.models import DualNet
from dn.trainer import Trainer
from dn.utils import deterministic, load_image_data_pickle, logger, metric

parser = argparse.ArgumentParser(description="DualNet-Image")
args = parser.parse_args()

if __name__ == "__main__":
    deterministic(args.seed)
    data = load_image_data_pickle(args.path)
    train_data, test_data = ImageDataDS(
        data.train_ds, transform=BarlowAugment()
    ), ImageDataDS(data.test_ds)
    train_data, test_data = l2l.data.MetaDataset(train_data), l2l.data.MetaDataset(
        test_data
    )

    transforms = [
        ContinousNWays(train_data, args.n_ways),
        l2l.data.transforms.LoadData(train_data),
    ]
    train_taskset = TaskDataset(
        train_data, transforms, num_tasks=args.n_class // args.n_ways
    )

    for epoch in range(args.epochs):
        for train_loader in MetaLoader(train_taskset, args, train=True):
            model = DualNet(args)
            trainer = Trainer(
                model,
                train_loader,
                test_loader,
                args.lr,
                args.epochs,
                metric=metric,
                logger=logger,
            )
            trainer.train()
            trainer.test()
            logger.checkpoint()
            logger.out()
