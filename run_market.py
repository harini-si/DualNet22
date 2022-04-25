import argparse
import logging
import time
from copy import deepcopy
from posixpath import split

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dn.data import (
    ContinousNWays,
    ImageData,
    ImageDataDS,
    MarketTaskset,
    MetaLoader,
    MyDS,
)
from dn.models import DualNetMarket
from dn.utils import deterministic, load_image_data_pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="./logs/log.txt",
    filemode="w+",
)


# default `log_dir` is "runs" - we'll be more specific here
parser = argparse.ArgumentParser(description="DualNet-Image")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_runs", type=int, default=5)
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--ssl_lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--path", type=str, default="./data/dfcl.csv")
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--input_dim", type=int, default=28)
parser.add_argument("--alpha", type=float, default=1e-3)
parser.add_argument("--out_dim", type=int, default=4)
parser.add_argument("--n_class", type=int, default=4)
parser.add_argument("--n_tasks", type=int, default=224)
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
parser.add_argument("--seq_len", type=int, default=5)
args = parser.parse_args()

if __name__ == "__main__":
    for run in args.n_runs:
        writer = SummaryWriter(f"results/test_VC_{time.time()}")
        deterministic(args)
        device = torch.device(args.device)

        train_taskset, test_taskset = MarketTaskset(
            "data/dfcl_mini.csv", args
        ), MarketTaskset("data/dfcl_mini.csv", args, split="Test")

        model = DualNetMarket(args).to(device)
        CLoss = torch.nn.CrossEntropyLoss()
        KLLoss = torch.nn.KLDivLoss()

        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        ssl_opt = torch.optim.SGD(model.SlowLearner.parameters(), lr=args.ssl_lr)
        for task, train_loader in enumerate(
            MetaLoader(train_taskset, args, train=True)
        ):
            logging.info("Running Task {}".format(task))
            model.train()
            if task > 0:
                x = model.memx[task]
                out = model(x)
                model.mem_feat[task] = F.softmax(out / model.temp, dim=1).data.clone()
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
                        SSL_loss = 0
                        for _ in range(args.inner_steps):
                            model.zero_grad()
                            if task > 0:
                                xx, yy, target, mask = model.memory_consolidation(task)
                                x1, x2 = model.augment(xx)
                            else:
                                x1, x2 = model.augment(x)
                            SSLLoss = model.SlowLearner((x1, x2))
                            SSLLoss.backward()
                            ssl_opt.step()
                            writer.add_scalar(
                                "SSL loss",
                                SSLLoss.item(),
                                epoch * len(train_loader) + i,
                            )

                        weights_after = model.state_dict()
                        new_params = {
                            name: weights_before[name]
                            + (
                                (weights_after[name] - weights_before[name])
                                * model.beta
                            )
                            for name in weights_before.keys()
                        }
                        model.load_state_dict(new_params)
                    running_loss = 0
                    running_loss1 = 0
                    correct = 0
                    total = 0
                    for _ in range(args.inner_steps):
                        model.zero_grad()
                        pred = model(x)
                        loss1 = CLoss(pred, y)
                        correct += torch.sum(torch.argmax(pred, dim=1) == y)
                        total += y.size(0) * y.size(-1)
                        writer.add_scalar(
                            "training acc",
                            correct.item() / total,
                            epoch * len(train_loader) + i,
                        )
                        writer.add_scalar(
                            "training loss", loss1.item(), epoch * len(train_loader) + i
                        )
                        loss2, loss3 = 0, 0
                        if task > 0:
                            xx, yy, target, mask = model.memory_consolidation(task)
                            pred = torch.gather(model(xx), 1, mask)
                            if (
                                (yy.detach().cpu().numpy().flatten() > 3).sum() > 0
                            ) or ((yy.detach().cpu().numpy().flatten() < 0).sum() > 0):
                                continue
                            loss2 += CLoss(pred, yy)
                            loss3 = model.reg * KLLoss(
                                F.log_softmax(pred / model.temp, dim=1), target
                            )
                        loss = loss1 + loss2 + loss3
                        writer.add_scalar(
                            "final loss", loss.item(), epoch * len(train_loader) + i
                        )
                        loss.backward()
                        opt.step()

            model.eval()
            mode = "test"
            for task_t, te_loader in enumerate(
                MetaLoader(test_taskset, args, train=False)
            ):
                if task_t > task:
                    break
                test_loss = 0
                correct, total = 0, 0
                for data, target in te_loader:
                    data, target = data.to(device), target.to(device)
                    logits = model(data)
                    loss = CLoss(logits, target)

                    writer.add_scalar(
                        "test loss", loss.item(), epoch * len(te_loader) + i
                    )
                    correct += torch.sum(torch.argmax(logits, dim=1) == target)
                    total += target.size(0) * target.size(-1)

                try:
                    acc = int(correct) / int(total)
                    logging.info("Task {} Acc: {:.4f}".format(task_t, acc))
                except ZeroDivisionError:
                    logging.info(f"Task {task_t} has 0 samples")
