import abc

import torch

from gans.gan_loss import GanLoss


class WganGpWithDriftLoss(GanLoss):
    __metaclass__ = abc.ABCMeta

    def __init__(self, grad_loss_weight: float = 10.0, drift_weight = 1e-3, device=torch.device('cpu')):
        super().__init__(device)
        self.grad_loss_weight = grad_loss_weight
        self.drift_weight = drift_weight

    def discriminator_loss(self,
                           G: torch.nn.Module,
                           D: torch.nn.Module,
                           real_image: torch.Tensor,
                           latent_vector: torch.Tensor) -> torch.Tensor:
        n = real_image.shape[0]
        assert latent_vector.shape[0] == n
        sample_size = 1
        for i in range(1,len(real_image.shape)):
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
        grad_diff = grad_norm - 1.0
        grad_loss = grad_diff.mul(grad_diff).mean() * self.grad_loss_weight

        D_real = D(real_image)
        real_loss = D_real.mean()
        fake_loss = D(fake_image).mean()
        drift_loss = D_real.mul(D_real).mean() * self.drift_weight

        return -real_loss + fake_loss + grad_loss + drift_loss

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
        return (combined_permuted[:n] * alpha + combined_permuted[n:] * (1 - alpha)).detach()

    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       real_image: torch.Tensor,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        return D(G(latent_vector)).mean() * -1.0


if __name__ == "__main__":
    wgan_gp_loss = WganGpWithDriftLoss()

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
