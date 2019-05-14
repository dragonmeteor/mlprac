import abc

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from gans.gan_loss import GanLoss


class RaSGanLoss(GanLoss):
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

        fake_image = G(latent_vector)

        C_real = D(real_image)
        C_fake = D(fake_image)

        D_real = C_real - C_fake.mean()
        D_fake = C_fake - C_real.mean()
        zeros = torch.zeros(n, 1, device=self.device, requires_grad=False)
        ones = torch.ones(n, 1, device=self.device, requires_grad=False)
        real_loss = binary_cross_entropy_with_logits(D_real, ones)
        fake_loss = binary_cross_entropy_with_logits(D_fake, zeros)

        sample_size = 1
        for i in range(1, len(real_image.shape)):
            sample_size *= real_image.shape[i]
        interpolates = self.create_interpolates(real_image, fake_image)
        interpolates.requires_grad_(True)
        grad_outputs = torch.ones([n, 1], device=self.device)
        D_interp = D(interpolates)
        interpolates_grad = torch.autograd.grad(D_interp,
                                                interpolates,
                                                grad_outputs=grad_outputs,
                                                only_inputs=True,
                                                create_graph=True,
                                                retain_graph=True)[0]
        grad_norm = interpolates_grad.view(n, sample_size).norm(2, dim=1)
        grad_loss = (grad_norm ** 2).mean() * self.grad_loss_weight

        return real_loss + fake_loss + grad_loss

    def create_interpolates(self,
                            real_image: torch.Tensor,
                            fake_image: torch.Tensor) -> torch.Tensor:
        n = real_image.shape[0]
        assert fake_image.shape[0] == n
        combined = torch.cat([real_image, fake_image], dim=0).detach()
        perm = torch.randperm(2 * n, device=self.device)
        combined_permuted = combined[perm, :]
        alpha_shape = [1 for i in range(len(real_image.shape))]
        alpha_shape[0] = n
        alpha = torch.rand(alpha_shape, device=self.device)
        return (combined_permuted[:n] * alpha
                + combined_permuted[n:] * (1 - alpha)).detach()


    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       real_image: torch.Tensor,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        n = real_image.shape[0]
        assert latent_vector.shape[0] == n

        fake_image = G(latent_vector)

        C_real = D(real_image)
        C_fake = D(fake_image)

        D_real = C_real - C_fake.mean()
        D_fake = C_fake - C_real.mean()
        zeros = torch.zeros(n, 1, device=self.device, requires_grad=False)
        ones = torch.ones(n, 1, device=self.device, requires_grad=False)
        real_loss = binary_cross_entropy_with_logits(D_real, zeros)
        fake_loss = binary_cross_entropy_with_logits(D_fake, ones)
        return real_loss + fake_loss
