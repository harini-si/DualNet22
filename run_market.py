import argparse
import logging
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dn.data import MarketTaskset, MetaLoader
from dn.models import DualNetMarket as DualNet
from dn.utils import VCMetrics, checkpoint, deterministic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename="./logs/log.txt",
    filemode="w+",
)

parser = argparse.ArgumentParser(description="DualNet-Market")
parser.add_argument("--path", type=str, default="data/dfcl_mini.csv")
parser.add_argument(
    "--save_path",
    type=str,
    default="results/",
    help="save models at the end of training",
)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--n_runs", type=int, default=5)
parser.add_argument("--n_epochs", type=int, default=2)
parser.add_argument("--inner_steps", type=int, default=1)
parser.add_argument("--n_outer", type=int, default=1)

parser.add_argument("--n_class", type=int, default=4)
parser.add_argument("--n_tasks", type=int, default=224)
parser.add_argument(
    "--n_memories", type=int, default=50, help="number of memories per task"
)
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--input_dim", type=int, default=28)
parser.add_argument("--out_dim", type=int, default=4)
parser.add_argument("--offsets", type=bool, default=False)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--ssl_lr", type=float, default=0.001)
parser.add_argument(
    "--memory_strength",
    default=1.0,
    type=float,
    help="memory strength (meaning depends on memory)",
)
parser.add_argument("--reg", default=1.0, type=float)
parser.add_argument(
    "--temp", type=float, default=1.0, help="temperature for distilation"
)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--ssl_loss", type=str, default="BarlowTwins")
args = parser.parse_args()

args.xdim = (args.seq_len, args.input_dim)
args.ydim = (args.out_dim,)

if __name__ == "__main__":
    metrics = VCMetrics(args)
    device = torch.device(args.device)
    train_taskset, test_taskset = MarketTaskset(args), MarketTaskset(args, split="Test")
    metrics.accs = np.zeros(args.n_runs, len(train_taskset), len(test_taskset))
    with tqdm(
        range(args.n_runs), desc="Runs Loop", leave=False, position=0, total=args.n_runs
    ) as pbar:
        for run in pbar:
            logging.info("Run {}".format(run))
            writer = SummaryWriter(f"{args.save_path}/test_MCL_{time.time()}")
            deterministic(args.seed + run)

            # create model and losses
            model = DualNet(args).to(device)
            CLoss = torch.nn.CrossEntropyLoss()
            KLLoss = torch.nn.KLDivLoss()

            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
            ssl_opt = torch.optim.SGD(model.SlowLearner.parameters(), lr=args.ssl_lr)

            with tqdm(
                enumerate(MetaLoader(train_taskset, args, train=True)),
                desc="Task Loop",
                total=args.n_tasks,
                leave=False,
                position=0,
            ) as outer:
                for task, train_loader in outer:
                    logging.info("Running Task {}".format(task))
                    model.train()
                    if task > 0:
                        model.memory.features_init(model, task - 1)

                    for epoch in range(args.n_epochs):
                        logging.info("Epoch {}".format(epoch))
                        with tqdm(
                            enumerate(train_loader),
                            desc="Train Loop",
                            total=len(train_loader),
                        ) as inner:
                            for i, (x, y) in inner:
                                x, y = x.to(device), y.to(device)
                                model.memory.update(x, y, task)

                                for j in range(args.n_outer):
                                    weights_before = deepcopy(model.state_dict())
                                    SSL_loss = 0
                                    for _ in range(args.inner_steps):
                                        model.zero_grad()
                                        if task > 0:
                                            (
                                                xx,
                                                yy,
                                                target,
                                                mask,
                                            ) = model.memory.consolidation(task)
                                            x1, x2 = model.barlow_augment(xx)
                                        else:
                                            x1, x2 = model.barlow_augment(x)
                                        SSLLoss = model.SlowLearner((x1, x2))
                                        SSLLoss.backward()
                                        ssl_opt.step()
                                        writer.add_scalar(
                                            "SSL loss",
                                            SSLLoss.item(),
                                            (epoch * len(train_loader) + i)
                                            * args.n_outer
                                            + j,
                                        )

                                    weights_after = model.state_dict()
                                    new_params = {
                                        name: weights_before[name]
                                        + (
                                            (weights_after[name] - weights_before[name])
                                            * args.beta
                                        )
                                        for name in weights_before.keys()
                                    }
                                    model.load_state_dict(new_params)
                                correct = 0
                                total = 0

                                for inner in range(args.inner_steps):
                                    model.zero_grad()
                                    pred = model(x)
                                    loss1 = CLoss(pred, y)
                                    correct += torch.sum(torch.argmax(pred, dim=1) == y)
                                    total += y.size(0) * y.size(-1)
                                    writer.add_scalar(
                                        "training acc",
                                        correct.item() / total,
                                        (epoch * len(train_loader) + i)
                                        * args.inner_steps
                                        + inner,
                                    )
                                    writer.add_scalar(
                                        "training loss",
                                        loss1.item(),
                                        (epoch * len(train_loader) + i)
                                        * args.inner_steps
                                        + inner,
                                    )
                                    loss2, loss3 = 0, 0
                                    if task > 0:
                                        (
                                            xx,
                                            yy,
                                            target,
                                            mask,
                                        ) = model.memory.consolidation(task)
                                        pred = torch.gather(model(xx), 1, mask)
                                        if (
                                            (
                                                yy.detach().cpu().numpy().flatten() > 3
                                            ).sum()
                                            > 0
                                        ) or (
                                            (
                                                yy.detach().cpu().numpy().flatten() < 0
                                            ).sum()
                                            > 0
                                        ):
                                            continue
                                        loss2 += CLoss(pred, yy)
                                        loss3 = args.reg * KLLoss(
                                            F.log_softmax(pred / args.temp, dim=1),
                                            target,
                                        )
                                    loss = loss1 + loss2 + loss3
                                    writer.add_scalar(
                                        "final loss",
                                        loss.item(),
                                        (epoch * len(train_loader) + i)
                                        * args.inner_steps
                                        + inner,
                                    )
                                    loss.backward()
                                    opt.step()

                    model.eval()
                    with tqdm(
                        enumerate(MetaLoader(test_taskset, args, train=False)),
                        desc="Test Loop",
                        total=task,
                        leave=False,
                        position=0,
                    ) as inner:
                        for task_t, te_loader in inner:
                            correct, total = 0, 0
                            for data, target in te_loader:
                                data, target = data.to(device), target.to(device)
                                logits = model(data)
                                correct += torch.sum(
                                    torch.argmax(logits, dim=1) == target
                                )
                                total += target.size(0) * target.size(-1)

                            try:
                                acc = correct / total
                            except ZeroDivisionError:
                                acc = np.nan
                                logging.info(f"Task {task_t} has 0 samples")
                            metrics.update_metric(run, task, task_t, acc)
                            logging.info("Task {} Acc: {:.4f}".format(task_t, acc))
            checkpoint(run, model, opt, ssl_opt, args)
    print(metrics)
    metrics.plot()
