import argparse
import pickle

import torch
from torch.utils.data import DataLoader

from dn.data import BarlowAugment, ImageData, ImageDataDS, MyDS
from dn.models import DualNet
from dn.trainer import Trainer
from dn.utils import deterministic, load_image_data_pickle, logger, metric

parser = argparse.ArgumentParser(description="DualNet-Image")
args = parser.parse_args()

if __name__ == "__main__":
    deterministic(args.seed)
    data = load_image_data_pickle(args.path)
    train_data = ImageDataDS(data.train_ds, transform=BarlowAugment())
    test_data = ImageDataDS(data.test_ds)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

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
