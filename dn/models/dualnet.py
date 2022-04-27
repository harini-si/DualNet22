import torch
from torch import nn

from dn.data import BarlowAugment, VCTransform
from dn.models.extractor import VCResNetFast, VCResNetSlow
from dn.models.memory import Memory

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
        # setup network
        super(DualNet, self).__init__()
        self.args = args
        self.n_class = args.n_class
        self.nc_per_task = int(args.n_class // args.n_tasks)

        # setup memories
        self.memory = Memory(args, self.nc_per_task, self.compute_offsets)

        # setup learners
        self.SlowLearner = SlowLearner(args)
        self.FastLearner = FastLearner(args)

        # Transforms
        self.VCTransform = VCTransform
        self.barlow_augment = BarlowAugment()

    def compute_offsets(self, task):
        return self.nc_per_task * task, self.nc_per_task * (task + 1)

    def forward(self, img, task, fast=False) -> torch.Tensor:
        """
        Fast Learner Inference
        """
        feat = self.SlowLearner(img, return_feat=True)
        out = self.FastLearner(img, feat)
        if fast:
            return out

        offset1, offset2 = self.compute_offsets(task)
        if offset1 > 0:
            out[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_class:
            out[:, int(offset2) : self.n_class].data.fill_(-10e10)

        return out
