import abc

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from gans.gan_loss import GanLoss


class SGanLoss(GanLoss):
    __metaclass__ = abc.ABCMeta

    def __init__(self, grad_loss_weight: float = 10.0, device=torch.device('cpu')):
        super().__init__(device)
        self.grad_loss_weight = grad_loss_weight

    def discriminator_loss(self,
                           G: torch.nn.Module,
                           D: torch.nn.Module,
                           real_image: torch.Tensor,
                           latent_vector: torch.Tensor) -> torch.Tensor:
        n = real_image.shape[0]
        assert latent_vector.shape[0] == n

        fake_image = G(latent_vector).detach()

        D_real = D(real_image)
        D_fake = D(fake_image)

        zeros = torch.zeros(n, 1, device=self.device, requires_grad=False)
        ones = torch.ones(n, 1, device=self.device, requires_grad=False)
        real_loss = binary_cross_entropy_with_logits(D_real, ones)
        fake_loss = binary_cross_entropy_with_logits(D_fake, zeros)

        return real_loss + fake_loss

    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       real_image: torch.Tensor,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        n = latent_vector.shape[0]
        assert latent_vector.shape[0] == n
        fake_image = G(latent_vector)
        D_fake = D(fake_image)
        ones = torch.ones(n, 1, device=self.device, requires_grad=False)
        fake_loss = binary_cross_entropy_with_logits(D_fake, ones)
        return fake_loss
