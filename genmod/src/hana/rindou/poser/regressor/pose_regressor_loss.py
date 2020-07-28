import abc

from torch import Tensor
from torch.nn import Module
from typing import List


class PoseRegressorLoss:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute(self, R: Module, batch: List[Tensor]) -> Tensor:
        pass