import torch
import abc


class MnistGanModule(torch.nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def initialize(self):
        pass


class MnistGan:
    __metaclass__ = abc.ABCMeta

    def __init__(self, device=torch.device('cpu')):
        self.image_size = 28
        self.image_vector_size = self.image_size * self.image_size
        self.latent_vector_size = 96
        self.default_batch_size = 128
        self.device = device

    @abc.abstractmethod
    def discriminator(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def generator(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def discriminator_loss(self,
                           D: torch.nn.Module,
                           real_image: torch.Tensor,
                           fake_image: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        pass
