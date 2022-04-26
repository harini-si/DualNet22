# necessary imports
import pickle
import random

import numpy as np
import seaborn as sns
import torch


def deterministic(args):
    # set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_image_data_pickle(path):
    # load image data from pickle file given path
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


class Metrics:
    def __init__(self, args):
        self.args = args
        self.accs = np.zeros((self.args.n_runs, self.args.n_tasks, self.args.n_tasks))

    def __repr__(self):
        final, best, forget, learn = self.get_metrics()
        return (
            f"Final Acc: {final}\n"
            f"Best Acc: {best}\n"
            f"Forgetting Measure: {forget}\n"
            f"Learn Acc: {learn}"
        )

    def update_metric(self, run, train_task, test_task, acc):
        self.accs[run][train_task][test_task] = acc

    def get_best_acc(self):
        return np.max(self.accs, 1)

    def get_learn_acc(self):
        np.diag(self.accs.mean(0))

    def get_final_acc(self):
        return self.accs[:, -1]

    def get_forgetting_measure(self, best, final):
        return best - final

    def get_metrics(self):
        final, best, learn = (
            self.get_final_acc(),
            self.get_best_acc(),
            self.get_learn_acc(),
        )
        return final, best, self.get_forgetting_measure(best, final), learn

    def plot(self, top_k=5):
        acc = self.acc.mean(0)[:, :top_k].T.reshape(
            -1,
        )
        train_task = np.concatenate([np.arange(self.args.n_tasks)] * top_k)
        test_task = np.arange(self.args.n_tasks * top_k) // self.args.n_tasks
        return sns.lineplot(x=train_task, y=acc, hue=test_task)
