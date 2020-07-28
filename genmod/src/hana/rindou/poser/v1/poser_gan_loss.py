import abc
from typing import List, Callable

from torch import Tensor
from torch.nn import Module


class PoserGanLoss:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def discriminator_loss(self,
                           G: Module,
                           D: Module,
                           batch: List[Tensor],
                           log_func: Callable[[str, float], None] = None) -> Tensor:
        pass

    @abc.abstractmethod
    def generator_loss(self,
                       G: Module,
                       D: Module,
                       batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        pass
