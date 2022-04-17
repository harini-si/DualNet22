import argparse

import torch

from dn.data import dataloader
from dn.models import DualNet
from dn.trainer import Trainer
from dn.utils import deterministic, logger, metric

args = argparse.parse_args()

if __name__ == "__main__":
    deterministic(args.seed)
    loader = dataloader(
        data=args.dataset, batch_size=args.batch_size, random_state=args.seed
    )
    model = DualNet(slow=args.slow_learner, fast=args.fast_learner)
    trainer = Trainer(model, loader, args.lr, args.epochs, metric=metric, logger=logger)
    trainer.train()
    trainer.test()
    logger.checkpoint()
    logger.out()
