import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Sequential, Conv2d, Tanh, BatchNorm1d, ReLU
from torch.nn.init import kaiming_normal_

from gans.common_layers import Unflatten, Flatten
from gans.gan_spec import Gan
from gans.resnet import ResidualBlock2d
from gans.spectral_norm_layers import SnConv2d, SnLinear

LATENT_VECTOR_SIZE = 256


class Resnet64Generator(Module):
    def __init__(self):
        super().__init__()
        self.first_linear = Linear(in_features=LATENT_VECTOR_SIZE, out_features=LATENT_VECTOR_SIZE * 4 * 4)
        kaiming_normal_(self.first_linear.weight)

        self.first_batchnorm = BatchNorm1d(num_features=LATENT_VECTOR_SIZE * 4 * 4)

        self.sequence = Sequential(
            Unflatten(channel=LATENT_VECTOR_SIZE, height=4, width=4),
            # 256 x 4 x 4
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True),
            # 256 x 4 x 4
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True),
            # 256 x 16 x 16
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True),
            # 256 x 16 x 16
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True),
            # 256 x 32 x 32
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True),
            # 256 x 32 x 32
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True),
            # 256 x 64 x 64
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True),
            # 256 x 64 x 64
            Conv2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=3,
                kernel_size=1),
            # 3 x 64 x 64
            Tanh())

    def forward(self, input):
        current = F.relu(self.first_batchnorm(self.first_linear(input)))
        return self.sequence(current)


class Resnet64Discriminator(Sequential):
    def __init__(self, use_spectral_normalization=False):
        if use_spectral_normalization:
            first_conv = SnConv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=1)
            last_linear = SnLinear(in_features=256 * 4 * 4, out_features=1)
        else:
            first_conv = Conv2d(
                in_channels=3,
                out_channels=256,
                kernel_size=1)
            last_linear = Linear(in_features=256 * 4 * 4, out_features=1)
        super().__init__(
            first_conv,
            # 256 x 64 x 64
            ReLU(),
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=False,
                use_spectral_normalization=use_spectral_normalization),
            # 256 x 32 x 32
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=False,
                use_spectral_normalization=use_spectral_normalization),
            # 256 x 16 x 16
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=False,
                use_spectral_normalization=use_spectral_normalization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                use_batchnorm=False,
                use_spectral_normalization=use_spectral_normalization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                use_batchnorm=False,
                use_spectral_normalization=use_spectral_normalization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=False,
                use_spectral_normalization=use_spectral_normalization),
            # 256 x 4 x 4
            Flatten(256 * 4 * 4),
            last_linear)


class Resnet64Gan(Gan):
    def __init__(self, use_spectral_normalization_in_discriminator=False):
        super().__init__()
        self.use_spectral_normalization_in_discriminator = use_spectral_normalization_in_discriminator

    @property
    def latent_vector_size(self) -> int:
        return LATENT_VECTOR_SIZE

    @property
    def image_size(self) -> int:
        return 64

    def generator(self) -> Module:
        return Resnet64Generator()

    def discriminator(self) -> Module:
        return Resnet64Discriminator(use_spectral_normalization=self.use_spectral_normalization_in_discriminator)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    G = Resnet64Generator().to(device=cuda)
    output = G(torch.zeros(64, 256, device=cuda))
    print(output.shape)

    D = Resnet64Discriminator().to(device=cuda)
    output2 = D(output)
    print(output2.shape)
