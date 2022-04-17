import torch


class SlowLearner(torch.nn.module):
    """
    Slow Learner Takes two images input and returns representation
    """

    def __init__(self, args):
        pass

    def forward(self, img, img_, *args, **kwargs) -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        pass

    def losser(self, img, img_, type="BarlowTwins") -> torch.Tensor:
        """
        Returns self sup loss
        """
        if type == "BarlowTwins":
            pass
        else:
            raise NotImplementedError


class FastLearner(torch.nn.module):
    """
    Fast Learner Takes image input and returns meta task output
    """

    def __init__(self, args):
        pass

    def forward(self, img, *args, **kwargs) -> torch.Tensor:
        """
        Obtain representation from slow learner
        """
        pass

    def losser(self, img) -> torch.Tensor:
        """
        Returns self sup loss
        """
        pass


class DualNet(torch.nn.module):
    """
    Takes Slow, fast learners and implements dualnet
    """

    def __init__(self, args):
        pass

    def forward(self, img, *args, **kwargs) -> torch.Tensor:
        """
        DualNet train step
        """
        pass
        self.memory_consolidation()
        self.feature_adaption()
        pass

    def memory_consolidation(self):
        raise NotImplementedError

    def feature_adaption(self):
        raise NotImplementedError
