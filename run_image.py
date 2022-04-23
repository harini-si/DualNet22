import argparse
import logging
from copy import deepcopy

import learn2learn as l2l
import torch
import torch.nn.functional as F
from learn2learn.data import TaskDataset

from dn.data import ContinousNWays, ImageData, ImageDataDS, MetaLoader, MyDS
from dn.models import DualNet
from dn.utils import deterministic, load_image_data_pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="./logs/log.txt",
    filemode="w+",
)
parser = argparse.ArgumentParser(description="DualNet-Image")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--ssl_lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--path", type=str, default="./data/image_data.pickle")
parser.add_argument("--n_ways", type=int, default=5)
parser.add_argument("--n_class", type=int, default=74)
parser.add_argument("--n_tasks", type=int, default=14)
parser.add_argument(
    "--n_memories", type=int, default=50, help="number of memories per task"
)
parser.add_argument(
    "--memory_strength",
    default=1.0,
    type=float,
    help="memory strength (meaning depends on memory)",
)
parser.add_argument("--reg", default=1.0, type=float)
parser.add_argument("--inner_steps", type=int, default=1)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="temperature for distilation"
)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument(
    "--save_path",
    type=str,
    default="results/",
    help="save models at the end of training",
)
parser.add_argument("--replay_batch_size", type=int, default=10)
parser.add_argument("--n_outer", type=int, default=1)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

if __name__ == "__main__":
    deterministic(args)
    device = torch.device(args.device)
    data = load_image_data_pickle(args.path)
    train_data, test_data = ImageDataDS(data.train_ds, transform=None), ImageDataDS(
        data.test_ds
    )
    train_data, test_data = l2l.data.MetaDataset(train_data), l2l.data.MetaDataset(
        test_data
    )
    logging.info("data loaded")
    transforms = [
        ContinousNWays(train_data, args.n_ways, args),
        l2l.data.transforms.LoadData(train_data),
    ]
    train_taskset = TaskDataset(
        train_data, transforms, num_tasks=args.n_class // args.n_ways
    )
    test_taskset = TaskDataset(
        test_data, transforms, num_tasks=args.n_class // args.n_ways
    )


    model = DualNet(args).to(device)
    CLoss = torch.nn.CrossEntropyLoss()
    KLLoss = torch.nn.KLDivLoss()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    ssl_opt = torch.optim.SGD(model.SlowLearner.parameters(), lr=args.ssl_lr)

    for task, train_loader in enumerate(MetaLoader(train_taskset, args, train=True)):
        logging.info("Running Task {}".format(task))
        model.train()
        if task > 0:
            # logging.info("Claculating Mem Features for task {}".format(task))
            offset1, offset2 = model.compute_offsets(task)
            x = model.VCTransform(model.memx[task])
            out = model(x, task)
            model.mem_feat[task] = F.softmax(
                out[:, offset1:offset2] / model.temp, dim=1
            ).data.clone()
        for epoch in range(args.n_epochs):
            logging.info("Epoch {}".format(epoch))
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                endcnt = min(model.mem_cnt + args.batch_size, model.n_memories)
                effbsz = endcnt - model.mem_cnt
                if effbsz > x.size(0):
                    effbsz = x.size(0)
                    endcnt = model.mem_cnt + effbsz
                model.memx[task, model.mem_cnt : endcnt].copy_(x.data[:effbsz])
                model.memy[task, model.mem_cnt : endcnt].copy_(y.data[:effbsz])
                model.mem_cnt += effbsz
                if model.mem_cnt == model.n_memories:
                    model.mem_cnt = 0

                for j in range(args.n_outer):
                    weights_before = deepcopy(model.state_dict())
                    for _ in range(args.inner_steps):
                        model.zero_grad()
                        if task > 0:
                            xx, yy, target, mask = model.memory_consolidation(task)
                            x1, x2 = model.barlow_augment(xx)
                        else:
                            x1, x2 = model.barlow_augment(x)
                        SSLLoss = model.SlowLearner((x1, x2))
                        SSLLoss.backward()
                        ssl_opt.step()

                    weights_after = model.state_dict()
                    new_params = {
                        name: weights_before[name]
                        + ((weights_after[name] - weights_before[name]) * model.beta)
                        for name in weights_before.keys()
                    }
                    model.load_state_dict(new_params)

                for _ in range(args.inner_steps):
                    model.zero_grad()
                    x = model.VCTransform(x)
                    offset1, offset2 = model.compute_offsets(task)
                    pred = model(x, task)
                    loss1 = CLoss(pred[:, offset1:offset2], y - offset1)
                    loss2, loss3 = 0, 0
                    if task > 0:
                        xx, yy, target, mask = model.memory_consolidation(task)
                        xx = model.VCTransform(xx)
                        pred = torch.gather(model(xx, task=None, fast=True), 1, mask)
                        loss2 += CLoss(pred, yy)
                        loss3 = model.reg * KLLoss(
                            F.log_softmax(pred / model.temp, dim=1), target
                        )
                    loss = loss1 + loss2 + loss3
                    loss.backward()
                    opt.step()
        model.eval()
        mode='test'
        for task_t, te_loader in enumerate(MetaLoader(test_taskset, args, train=False)):
            if task_t > task: break
            
            for data, target in te_loader:
                data, target = data.cuda(), target.cuda()
                logits = model(data, task_t)
                loss = F.cross_entropy(logits, target)
                pred = logits.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(data)
                logging.info("Task {} Acc: {:.4f}".format(task_t, acc))