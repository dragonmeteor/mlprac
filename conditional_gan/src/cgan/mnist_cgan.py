from typing import Callable

import abc
import torch
from torch import Tensor

class MnistCgan(abc.ABC):
    LATENT_VECTOR_SIZE = 100
    IMAGE_SIZE = 28
    IMAGE_VECTOR_SIZE = 784

    def __init__(self, device=torch.device('cuda')):
        self.device = device

    @abc.abstractmethod
    def discriminator(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def generator(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def discriminator_loss(self) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @abc.abstractmethod
    def generator_loss(self) -> Callable[[Tensor, Tensor], Tensor]:
        pass
