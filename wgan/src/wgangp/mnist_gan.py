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

    IMAGE_SIZE = 28
    IMAGE_VECTOR_SIZE = 28 * 28
    LATENT_VECTOR_SIZE = 96
    DEFAULT_BATCH_SIZE = 128

    def __init__(self, device=torch.device('cpu')):
        self.device = device

    @abc.abstractmethod
    def discriminator(self) -> MnistGanModule:
        pass

    @abc.abstractmethod
    def generator(self) -> MnistGanModule:
        pass

    @abc.abstractmethod
    def discriminator_loss(self,
                           G: torch.nn.Module,
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
