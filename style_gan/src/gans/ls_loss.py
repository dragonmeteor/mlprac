import torch

from gans.gan_loss import GanLoss


class LsLoss(GanLoss):
    def __init__(self, device=torch.device('cuda')):
        super().__init__(device)

    def discriminator_loss(self,
                           G: torch.nn.Module,
                           D: torch.nn.Module,
                           real_image: torch.Tensor,
                           latent_vector: torch.Tensor) -> torch.Tensor:
        fake_image = G(latent_vector).detach()
        fake_logit = D(fake_image)
        fake_loss = fake_logit.mul(fake_logit).mean() / 2.0

        real_logit = D(real_image)
        real_diff = real_logit - 1.0
        real_loss = real_diff.mul(real_diff).mean() / 2.0

        return real_loss + fake_loss

    def generator_loss(self,
                       G: torch.nn.Module,
                       D: torch.nn.Module,
                       latent_vector: torch.Tensor) -> torch.Tensor:
        fake_logit = D(G(latent_vector))
        fake_diff = fake_logit - 1.0
        return fake_diff.mul(fake_diff).mean() / 2.0