import abc

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from gans.gan_loss import GanLoss


class ZeroGpLoss(GanLoss):
    __metaclass__ = abc.ABCMeta

    def __init__(self, grad_loss_weight: float = 10.0, device=torch.device('cpu')):
        super().__init__(device)
        self.grad_loss_weight = grad_loss_weight

    def discriminator_loss(self,
                           G,
                           D: torch.nn.Module,
                           real_image: torch.Tensor,
                           latent_vector: torch.Tensor) -> torch.Tensor:
        n = real_image.shape[0]
        assert latent_vector.shape[0] == n
        sample_size = 1
        for i in range(1, len(real_image.shape)):
            sample_size *= real_image.shape[i]

        fake_image = G(latent_vector).detach()

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

        real_logit = D(real_image)
        fake_logit = D(fake_image)
        zeros = torch.zeros(n, 1, device=self.device, requires_grad=False)
        ones = torch.ones(n, 1, device=self.device, requires_grad=False)
        real_loss = binary_cross_entropy_with_logits(real_logit, ones)
        fake_loss = binary_cross_entropy_with_logits(fake_logit, zeros)

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
                       G,
                       D: torch.nn.Module,
                       real_image: torch.Tensor,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        fake_logit = D(G(latent_vector))
        ones = torch.ones(latent_vector.shape[0], 1, device=self.device, requires_grad=False)
        return binary_cross_entropy_with_logits(fake_logit, ones)

