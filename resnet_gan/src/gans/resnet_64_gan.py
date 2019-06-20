import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Sequential, Conv2d, Tanh, BatchNorm1d, ReLU, BatchNorm2d
from torch.nn.init import kaiming_normal_, xavier_normal_
from torch.nn.utils.spectral_norm import SpectralNorm, spectral_norm

from gans.common_layers import Unflatten, Flatten
from gans.gan_spec import Gan
from gans.resnet import ResidualBlock2d

LATENT_VECTOR_SIZE = 256


class Resnet64Generator(Module):
    def __init__(self, initialization='he'):
        super().__init__()
        self.first_linear = Linear(in_features=LATENT_VECTOR_SIZE, out_features=LATENT_VECTOR_SIZE * 4 * 4)
        if initialization == "he":
            kaiming_normal_(self.first_linear.weight)
        else:
            xavier_normal_(self.first_linear.weight)

        self.first_batchnorm = BatchNorm1d(num_features=LATENT_VECTOR_SIZE * 4 * 4)

        last_conv2d = Conv2d(
            in_channels=LATENT_VECTOR_SIZE,
            out_channels=3,
            kernel_size=1)
        if initialization == "he":
            kaiming_normal_(last_conv2d.weight)
        else:
            xavier_normal_(last_conv2d.weight)

        self.sequence = Sequential(
            Unflatten(channel=LATENT_VECTOR_SIZE, height=4, width=4),
            # 256 x 4 x 4
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 4 x 4
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 16 x 16
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 16 x 16
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 32 x 32
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 32 x 32
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                resample="up",
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 64 x 64
            ResidualBlock2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=LATENT_VECTOR_SIZE,
                use_batchnorm=True,
                initialization=initialization),
            # 256 x 64 x 64
            last_conv2d,
            # 3 x 64 x 64
            Tanh())

    def forward(self, input):
        current = F.relu(self.first_batchnorm(self.first_linear(input)))
        return self.sequence(current)


class Resnet64Discriminator(Sequential):
    def __init__(self, use_spectral_normalization=False, use_batchnorm=False, initialization="he"):
        first_conv = Conv2d(
            in_channels=3,
            out_channels=256,
            kernel_size=1)
        last_linear = Linear(in_features=256 * 4 * 4, out_features=1)

        if initialization == "he":
            kaiming_normal_(first_conv.weight)
            kaiming_normal_(last_linear.weight)
        else:
            xavier_normal_(first_conv.weight)
            xavier_normal_(last_linear.weight)

        if use_spectral_normalization:
            first_conv = spectral_norm(first_conv)
            last_linear = spectral_norm(last_linear)

        if use_batchnorm:
            first_conv = Sequential(
                first_conv,
                BatchNorm2d(num_features=256))

        super().__init__(
            first_conv,
            # 256 x 64 x 64
            ReLU(),
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 32 x 32
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 16 x 16
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 8 x 8
            ResidualBlock2d(
                in_channels=256,
                out_channels=256,
                resample="down",
                use_batchnorm=use_batchnorm,
                use_spectral_normalization=use_spectral_normalization,
                initialization=initialization),
            # 256 x 4 x 4
            Flatten(256 * 4 * 4),
            last_linear)


class Resnet64Gan(Gan):
    def __init__(self,
                 use_spectral_normalization_in_discriminator=False,
                 use_batchnorm_in_discriminator=False,
                 initialization="he"):
        super().__init__()
        self.use_spectral_normalization_in_discriminator = use_spectral_normalization_in_discriminator
        self.use_bathnorm_in_discriminator = use_batchnorm_in_discriminator
        self.initialization = initialization

    @property
    def latent_vector_size(self) -> int:
        return LATENT_VECTOR_SIZE

    @property
    def image_size(self) -> int:
        return 64

    def generator(self) -> Module:
        return Resnet64Generator(initialization=self.initialization)

    def discriminator(self) -> Module:
        return Resnet64Discriminator(
            use_spectral_normalization=self.use_spectral_normalization_in_discriminator,
            use_batchnorm=self.use_bathnorm_in_discriminator,
            initialization=self.initialization)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    G = Resnet64Generator().to(device=cuda)
    output = G(torch.zeros(64, 256, device=cuda))
    print(output.shape)

    D = Resnet64Discriminator().to(device=cuda)
    output2 = D(output)
    print(output2.shape)
