import abc

import torch

from gans.gan_loss import GanLoss


class WganGpLoss(GanLoss):
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

        interpolates = self.create_interpolates(real_image, fake_image)
        interpolates.requires_grad_(True)
        grad_outputs = torch.ones([n, 1], device=self.device)
        interpolates_grad = torch.autograd.grad(D(interpolates),
                                                interpolates,
                                                grad_outputs=grad_outputs,
                                                only_inputs=True,
                                                create_graph=True,
                                                retain_graph=True)[0]
        grad_norm = interpolates_grad.norm(2, dim=1)
        grad_diff = grad_norm - 1.0
        grad_loss = grad_diff.mul(grad_diff).mean() * self.grad_loss_weight

        real_loss = D(real_image).mean()
        fake_loss = D(fake_image).mean()

        return -real_loss + fake_loss + grad_loss

    def create_interpolates(self,
                            real_image: torch.Tensor,
                            fake_image: torch.Tensor) -> torch.Tensor:
        n = real_image.shape[0]
        combined = torch.cat([real_image, fake_image], dim=0).detach()
        perm = torch.randperm(2 * n, device=self.device)
        combined_permuted = combined[perm, :]
        alpha = torch.rand([n, 1], device=self.device)
        return (combined_permuted[:n, :] * alpha
                + combined_permuted[n:, :] * (1 - alpha)).detach()

    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        return D(G(latent_vector)).mean() * -1.0


if __name__ == "__main__":
    wgan_gp_loss = WganGpLoss()

    real_image = torch.Tensor([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])
    fake_image = torch.Tensor([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    print(wgan_gp_loss.create_interpolates(real_image, fake_image))
