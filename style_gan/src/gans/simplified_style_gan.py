import torch

from torch.nn import Module, Sequential, Linear, LeakyReLU, Conv2d, Parameter
from torch.nn import functional as F
from torch.nn.init import ones_, zeros_

from gans.gan_module import GanModule

LATENT_VECTOR_SIZE = 512
LEAKY_RELU_NEGATIVE_SLOPE = 0.2


class MappingNetwork(GanModule):
    def __init__(self,
                 latent_vector_size: int = LATENT_VECTOR_SIZE,
                 leaky_lu_negative_slope: float = LEAKY_RELU_NEGATIVE_SLOPE):
        super().__init__()
        self.sequential = Sequential(
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope),
            Linear(in_features=latent_vector_size, out_features=latent_vector_size),
            LeakyReLU(negative_slope=leaky_lu_negative_slope))

    def forward(self, input):
        return self.sequential(input)

    def initialize(self):
        pass


def AdaIN(input: torch.Tensor, new_mean: torch.Tensor, new_std: torch.Tensor, epsilon: float = 1e-8):
    n = input.shape[0]
    c = input.shape[1]
    h = input.shape[2]
    w = input.shape[2]

    input_flattened = input.view(n, c, h * w)
    input_mean = input_flattened.mean(dim=2, keepdim=True)
    input_std = (input_flattened.var(dim=2, keepdim=True) + epsilon).sqrt()
    input_normalized = (input_flattened - input_mean) / input_std

    return (input_normalized * new_std.view(n, c, 1) + new_mean.view(n, c, 1)).view(n, c, h, w)


class GeneratorBlock(GanModule):
    def __init__(self, in_channels: int, out_channels: int,
                 latent_vector_size: int = LATENT_VECTOR_SIZE,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_vector_size = latent_vector_size
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

        self.conv_1 = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

        self.noise_1_factor = Parameter(torch.Tensor(1, out_channels, 1, 1))
        zeros_(self.noise_1_factor)

        self.weight_to_new_mean_1 = Linear(in_features=latent_vector_size, out_features=out_channels)
        self.weight_to_new_std_1 = Linear(in_features=latent_vector_size, out_features=out_channels)

        self.conv_2 = Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

        self.noise_2_factor = Parameter(torch.Tensor(1, out_channels, 1, 1))
        zeros_(self.noise_2_factor)

        self.weight_to_new_mean_2 = Linear(in_features=latent_vector_size, out_features=out_channels)
        self.weight_to_new_std_2 = Linear(in_features=latent_vector_size, out_features=out_channels)

    def upsample(self, input_image):
        return F.interpolate(input_image, scale_factor=2, mode='nearest')

    def forward(self,
                input_image: torch.Tensor,
                weight: torch.Tensor,
                noise_1: torch.Tensor = None,
                noise_2: torch.Tensor = None):
        upsampled_image = self.upsample(input_image)
        conv_1_image = self.convolve_1(upsampled_image)
        noise_1_image = conv_1_image + self.create_noise(conv_1_image, self.noise_1_factor, noise_1)
        adain_1_image = AdaIN(noise_1_image, self.weight_to_new_mean_1(weight), self.weight_to_new_std_1(weight))
        conv_2_image = self.convolve_2(adain_1_image)
        noise_2_image = conv_2_image + self.create_noise(conv_2_image, self.noise_2_factor, noise_2)
        adain_2_image = AdaIN(noise_2_image, self.weight_to_new_mean_2(weight), self.weight_to_new_std_2(weight))
        return adain_2_image

    def convolve_1(self, input_image: torch.Tensor):
        return F.leaky_relu(self.conv_1(input_image), negative_slope=self.leaky_relu_negative_slope)

    def convolve_2(self, input_image: torch.Tensor):
        return F.leaky_relu(self.conv_2(input_image), negative_slope=self.leaky_relu_negative_slope)

    def create_noise(self, input_image: torch.Tensor, noise_factor: torch.Tensor, input_noise: torch.Tensor = None):
        n = input_image.shape[0]
        c = input_image.shape[1]
        h = input_image.shape[2]
        w = input_image.shape[3]

        if input_noise is None:
            raw_noise = torch.randn(n, 1, h, w).expand(n, c, h, w)
        else:
            raw_noise = input_noise.view(1,1,h,w).expand(n, c, h, w)
        return raw_noise * noise_factor

    def initialize(self):
        pass