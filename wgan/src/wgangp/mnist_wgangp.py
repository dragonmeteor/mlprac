import abc
import torch

from wgangp.mnist_gan import MnistGan


class MnistWganGp(MnistGan):
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
        assert real_image.shape == torch.Size([n, MnistGan.IMAGE_VECTOR_SIZE])
        assert latent_vector.shape == torch.Size([n, MnistGan.LATENT_VECTOR_SIZE])

        fake_image = G(latent_vector).detach()

        interpolates = self.create_interpolates(real_image, fake_image)
        interpolates.requires_grad_(True)
        interpolates_grad = torch.zeros([n, MnistGan.IMAGE_VECTOR_SIZE], device=self.device)
        torch.autograd.grad(D(interpolates),
                            interpolates,
                            grad_outputs=interpolates_grad,
                            only_inputs=True,
                            create_graph=True)
        grad_loss = ((interpolates_grad - 1.0) ** 2).mean() * self.grad_loss_weight

        real_loss = D(real_image).mean()
        fake_loss = D(fake_image).mean() * -1

        return real_loss - fake_loss + grad_loss

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
