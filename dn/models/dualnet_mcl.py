import torch
from torch import nn

from dn.data import Corrupt
from dn.models.extractor import LSTMFast, LSTMSlow

n_class = 74
SLOW_EMBEDDER = LSTMSlow
FAST_EMBEDDER = LSTMFast


class SlowLearnerMarket(torch.nn.Module):
    """
    Slow Learner Takes two images input and returns representation
    """

    def __init__(self, args):
        super(SlowLearnerMarket, self).__init__()
        self.args = args
        self.embedder = SLOW_EMBEDDER(args)
        self.augmenter = Corrupt()

    def forward(self, input, type="BarlowTwins", return_feat=False) -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        if return_feat:
            feat = self.embedder(input, return_feat=True)
            return feat

        else:
            emb, emb_ = self.embedder(input[0], return_feat=False), self.embedder(
                input[1], return_feat=False
            )
            if type == "BarlowTwins":
                return self.barlow_twins_loss(emb, emb_)
            else:
                raise NotImplementedError

    def barlow_twins_loss(self, z1, z2):
        """
        Input: z1, z2 embeddings
        Returns loss by barlo twins method
        """
        z_a = (z1 - z1.mean(0)) / (z1.std(0) + self.args.alpha)
        z_b = (z2 - z2.mean(0)) / (z2.std(0) + self.args.alpha)
        N, D = z_a.size(0), z_a.size(1)
        c_ = torch.mm(z_a.T, z_b) / N
        diag = torch.eye(D).to(self.args.device)
        c_diff = (c_ - diag).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
        loss = c_diff.sum()
        return loss


class FastLearnerMarket(torch.nn.Module):
    """
    Fast Learner Takes image input and returns meta task output
    """

    def __init__(self, args):
        super(FastLearnerMarket, self).__init__()
        self.args = args
        self.embedder = FAST_EMBEDDER(args)

    def forward(self, stock, feat) -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        out = self.embedder(stock, feat)
        return out


class DualNetMarket(torch.nn.Module):
    """
    Takes Slow, fast learners and implements dualnet
    """

    def __init__(self, args):
        super(DualNetMarket, self).__init__()
        self.args = args
        self.reg = args.memory_strength
        self.temp = args.temperature
        self.beta = args.beta

        # setup memories
        self.n_memories = args.n_memories
        self.mem_cnt = 0
        self.memx = torch.FloatTensor(
            args.n_tasks, self.n_memories, args.seq_len, args.input_dim
        ).to(self.args.device)
        self.memy = torch.LongTensor(args.n_tasks, self.n_memories, args.out_dim).to(
            self.args.device
        )
        self.mem_feat = torch.FloatTensor(
            args.n_tasks, self.n_memories, args.out_dim, args.n_class
        ).to(self.args.device)
        self.mem = {}

        self.bsz = args.batch_size
        self.n_class = args.n_class
        self.rsz = args.replay_batch_size

        self.inner_steps = args.inner_steps
        self.n_outer = args.n_outer

        self.SlowLearner = SlowLearnerMarket(args)
        self.FastLearner = FastLearnerMarket(args)

        self.augment = Corrupt()

    def forward(self, stock) -> torch.Tensor:
        """
        Fast Learner Inference
        """
        feat = self.SlowLearner(stock, return_feat=True)
        out = self.FastLearner(stock, feat)
        return out

    def memory_consolidation(self, task):
        t = torch.randint(0, task, (self.bsz,))
        x = torch.randint(0, self.n_memories, (self.bsz,))
        xx = self.memx[t, x]
        yy = self.memy[t, x]
        feat = self.mem_feat[t, x]
        mask = torch.zeros(self.bsz, self.args.out_dim, self.args.n_class)
        for j in range(self.bsz):
            mask[j] = torch.stack(self.args.out_dim * [torch.arange(self.args.n_class)])
        return (
            xx.to(self.args.device),
            yy.to(self.args.device),
            feat.to(self.args.device),
            mask.long().to(self.args.device),
        )
