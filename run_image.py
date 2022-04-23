import argparse
import pickle
from copy import deepcopy

import learn2learn as l2l
import torch
import torch.nn.Functional as F
from learn2learn.data import TaskDataset
from torch.utils.data import DataLoader

from dn.data import ContinousNWays, ImageData, ImageDataDS, MetaLoader, MyDS
from dn.models import DualNet
from dn.trainer import Trainer
from dn.utils import deterministic, load_image_data_pickle, logger, metric

parser = argparse.ArgumentParser(description="DualNet-Image")
args = parser.parse_args()

if __name__ == "__main__":
    deterministic(args.seed)
    data = load_image_data_pickle(args.path)
    train_data, test_data = ImageDataDS(data.train_ds, transform=None), ImageDataDS(
        data.test_ds
    )
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

    model = DualNet(args)
    CLoss = torch.nn.CrossEntropyLoss()
    KLLoss = torch.nn.KLDivLoss()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    ssl_opt = torch.optim.SGD(model.SlowLearner.parameters(), lr=args.bt_lr)

    for task, train_loader in enumerate(MetaLoader(train_taskset, args, train=True)):
        model.train()
        if task > 0:
            offset1, offset2 = model.compute_offsets(task)
            x = model.vctransform(model.memx[task])
            out = model(x, task)
            model.mem_feat[task] = F.softmax(
                out[:, offset1:offset2] / model.temp, dim=1
            ).data.clone()
        for _ in range(args.n_epochs):
            for x, y in train_loader:
                endcnt = min(model.mem_cnt + args.batch_size, model.n_memories)
                model.memx[task, model.mem_cnt : endcnt].copy_(
                    x.data[: args.batch_size]
                )
                model.memy[task, model.mem_cnt : endcnt].copy_(
                    y.data[: args.batch_size]
                )
                model.mem_cnt += args.batch_size

                if model.mem_cnt == model.n_memories:
                    model.mem_cnt = 0

                for j in range(args.n_outer):
                    weights_before = deepcopy(model.net.state_dict())
                    for _ in range(args.inner_steps):
                        model.zero_grad()
                        if task > 0:
                            xx, yy, target, mask = model.memory_sampling(
                                task, args.batch_size
                            )
                            x1, x2 = model.barlow_augment(xx)
                        else:
                            x1, x2 = model.barlow_augment(x)

                        SSLLoss = model.SlowLearner(x1, x2)
                        SSLLoss.backward()
                        ssl_opt.step()

                    weights_after = model.net.state_dict()
                    new_params = {
                        name: weights_before[name]
                        + ((weights_after[name] - weights_before[name]) * model.beta)
                        for name in weights_before.keys()
                    }
                    model.net.load_state_dict(new_params)

                for _ in range(args.inner_steps):
                    model.zero_grad()
                    x = model.vctransform(x)
                    offset1, offset2 = model.compute_offsets(task)
                    pred = model(x, task, True)
                    loss1 = CLoss(pred[:, offset1:offset2], y - offset1)
                    loss2 = 0
                    if task > 0:
                        xx, yy, target, mask = model.memory_sampling(task)
                        xx = model.vctransform(xx)
                        pred = torch.gather(model(xx), 1, mask)
                        loss2 += CLoss(pred, yy)
                        loss3 = model.reg * KLLoss(
                            F.log_softmax(pred / model.temp, dim=1), target
                        )
                    loss = loss1 + loss2 + loss3
                    loss.backward()
                    opt.step()
