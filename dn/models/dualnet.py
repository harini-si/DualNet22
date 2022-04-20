import torch
from torch import nn

from dn.data import BarlowAugment
from dn.models.cnn import CNN

n_class = 74
SLOW_EMBEDDER = CNN
FAST_EMBEDDER = CNN


class SlowLearner(torch.nn.Module):
    """
    Slow Learner Takes two images input and returns representation
    """

    def __init__(self, args):
        super(SlowLearner, self).__init__()
        self.args = args
        self.embedder = SLOW_EMBEDDER(args.n_class)
        self.augmenter = BarlowAugment()
        pass

    def forward(self, img, img_, type="BarlowTwins") -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        if type == "BarlowTwins":
            img, img_ = self.embedder(img), self.embedder(img_)
            return self.barlow_twins_losser(img, img_)
        else:
            raise NotImplementedError
        pass

    def barlow_twins_losser(self, z1, z2):
        """
        Input: z1, z2 embeddings
        Returns loss by barlo twins method
        """
        z_a = (z1 - z1.mean(0)) / z1.std(0)
        z_b = (z2 - z2.mean(0)) / z2.std(0)
        N, D = z_a.size(0), z_a.size(1)
        c_ = torch.mm(z_a.T, z_b) / N
        diag = torch.eye(D)
        if self.args.device == "cuda":
            diag = diag.cuda()
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
        self.embedder = FAST_EMBEDDER(args.n_class)
        self.fc = nn.Linear(args.emb, args.n_class)

    def forward(self, img) -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        emb = self.embedder(img)
        out = self.fc(emb)
        return out


class DualNet(torch.nn.Module):
    """
    Takes Slow, fast learners and implements dualnet
    """

    def __init__(self, args):
        super(DualNet, self).__init__()
        self.SlowLearner = SlowLearner(args)
        self.FastLearner = FastLearner(args)
        # self.mem=
        pass

    def forward(self, img, *args, **kwargs) -> torch.Tensor:
        """
        DualNet train step
        """
        pass
        self.memory_consolidation()
        self.feature_adaption()
        pass

    def memory_consolidation(self, img, label):

        raise NotImplementedError

    def feature_adaption(self):
        raise NotImplementedError
