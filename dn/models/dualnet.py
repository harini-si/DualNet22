import torch
from torch import nn

from dn.data import BarlowAugment, VCTransform
from dn.models.extractor import VCResNetFast, VCResNetSlow

n_class = 74
SLOW_EMBEDDER = VCResNetSlow
FAST_EMBEDDER = VCResNetFast


class SlowLearner(torch.nn.Module):
    """
    Slow Learner Takes two images input and returns representation
    """

    def __init__(self, args):
        super(SlowLearner, self).__init__()
        self.args = args
        self.embedder = SLOW_EMBEDDER(args)
        self.augmenter = BarlowAugment()

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
                return self.barlow_twins_losser(emb, emb_)
            else:
                raise NotImplementedError

    def barlow_twins_losser(self, z1, z2):
        """
        Input: z1, z2 embeddings
        Returns loss by barlo twins method
        """
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)
        N, D = z_a.size(0), z_a.size(1)
        c_ = torch.mm(z_a.T, z_b) / N
        diag = torch.eye(D).to(self.args.device)
        c_diff = (c_ - diag).pow(2)
        c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
        loss = c_diff.sum()
        return loss


class FastLearner(torch.nn.Module):
    """
    Fast Learner Takes image input and returns meta task output
    """

    def __init__(self, args):
        super(FastLearner, self).__init__()
        self.args = args
        self.embedder = FAST_EMBEDDER(args)

    def forward(self, img, feat) -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        out = self.embedder(img, feat)
        return out


class DualNet(torch.nn.Module):
    """
    Takes Slow, fast learners and implements dualnet
    """

    def __init__(self, args):
        super(DualNet, self).__init__()
        self.reg = args.memory_strength
        self.temp = args.temperature
        self.beta = args.beta
        self.nc_per_task = int(args.n_class // args.n_tasks)

        # setup memories
        self.n_memories = args.n_memories
        self.mem_cnt = 0
        self.memx = torch.FloatTensor(args.n_tasks, self.n_memories, 1, 20, 20).to(
            self.args.device
        )
        self.memy = torch.LongTensor(args.n_tasks, self.n_memories).to(self.args.device)
        self.mem_feat = torch.FloatTensor(
            args.n_tasks, self.n_memories, self.nc_per_task
        ).to(self.args.device)
        self.mem = {}

        self.bsz = args.batch_size
        self.n_class = args.n_class
        self.rsz = args.replay_batch_size

        self.inner_steps = args.inner_steps
        self.n_outer = args.n_outer

        self.SlowLearner = SlowLearner(args)
        self.FastLearner = FastLearner(args)

        self.VCTransform = VCTransform
        self.barlow_augment = BarlowAugment()

    def compute_offsets(self, task):
        return self.nc_per_task * task, self.nc_per_task * (task + 1)

    def forward(self, img, task) -> torch.Tensor:
        """
        Fast Learner Inference
        """
        feat = self.SlowLearner(img, return_feat=True)
        out = self.FastLearner(img, feat)

        offset1, offset2 = self.compute_offsets(task)
        if offset1 > 0:
            out[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_class:
            out[:, int(offset2) : self.n_class].data.fill_(-10e10)

        return out

    def memory_consolidation(self, task):
        t = torch.randint(0, task, (self.bsz,))
        x = torch.randint(0, self.n_memories, (self.bsz,))
        offsets = torch.tensor([self.compute_offsets(i) for i in t])
        xx = self.memx[t, x]
        yy = self.memy[t, x] - offsets[:, 0]
        feat = self.mem_feat[t, x]
        mask = torch.zeros(self.bsz, self.nc_per_task)
        for j in range(self.bsz):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        return (
            xx.to(self.args.device),
            yy.to(self.args.device),
            feat.to(self.args.device),
            mask.long().to(self.args.device),
        )
