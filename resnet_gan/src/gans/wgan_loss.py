import abc

import torch

from gans.gan_loss import GanLoss


class WganLoss(GanLoss):
    __metaclass__ = abc.ABCMeta

    def __init__(self, device=torch.device('cpu')):
        super().__init__(device)

    def discriminator_loss(self,
                           G: torch.nn.Module,
                           D: torch.nn.Module,
                           real_image: torch.Tensor,
                           latent_vector: torch.Tensor) -> torch.Tensor:
        fake_image = G(latent_vector).detach()

        D_real = D(real_image)
        real_loss = D_real.mean()
        fake_loss = D(fake_image).mean()

        return -real_loss + fake_loss

    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       real_image: torch.Tensor,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        return D(G(latent_vector)).mean() * -1.0