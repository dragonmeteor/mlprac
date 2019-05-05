import torch
import abc


class GanModule(torch.nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def initialize(self):
        pass


class Gan:
    __metaclass__ = abc.ABCMeta

    def __init__(self, device=torch.device('cpu')):
        self.device = device

    @property
    @abc.abstractmethod
    def sample_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def latent_vector_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def image_size(self) -> int:
        pass

    @abc.abstractmethod
    def discriminator(self) -> GanModule:
        pass

    @abc.abstractmethod
    def generator(self) -> GanModule:
        pass
