import abc
import torch


class GanLoss:
    __metaclass__ = abc.ABCMeta

    def __init__(self, device=torch.device("cpu")):
        self.device = device

    @abc.abstractmethod
    def discriminator_loss(self,
                           G: torch.nn.Module,
                           D: torch.nn.Module,
                           real_image: torch.Tensor,
                           latent_vector: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       real_imge: torch.Tensor,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        pass