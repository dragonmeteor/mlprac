import torch
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, Linear, ReLU, ConvTranspose2d, BatchNorm2d, Tanh, MaxPool2d, \
    BatchNorm1d
from torch.nn.init import xavier_uniform_

from wgangp.gan import Gan, GanModule


class Flatten(Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input: torch.Tensor):
        return input.view(input.shape[0], self.size)


class Unflatten(Module):
    def __init__(self, channel, height, width):
        super().__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input: torch.Tensor):
        return input.view(input.shape[0], self.channel, self.height, self.width)


IMAGE_SIZE = 28
SAMPLE_SIZE = 28 * 28
LATENT_VECTOR_SIZE = 100


class MnistDcGanDiscriminator(GanModule):
    def __init__(self, leaky_relu_negative_slope=0.01):
        super().__init__()
        self.sequential = Sequential(
            Unflatten(1, IMAGE_SIZE, IMAGE_SIZE),
            Conv2d(in_channels=1,
                   out_channels=32,
                   kernel_size=5,
                   stride=1),
            LeakyReLU(negative_slope=leaky_relu_negative_slope),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=32,
                   out_channels=64,
                   kernel_size=5,
                   stride=1),
            LeakyReLU(negative_slope=leaky_relu_negative_slope),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(1024),
            Linear(in_features=1024, out_features=1024),
            LeakyReLU(negative_slope=leaky_relu_negative_slope),
            Linear(in_features=1024, out_features=1))

    def forward(self, x):
        return self.sequential(x)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, Linear) or isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                xavier_uniform_(module.weight)


class MnistDcGanGenerator(GanModule):
    def __init__(self):
        super().__init__()
        self.sequential = Sequential(
            Linear(in_features=LATENT_VECTOR_SIZE, out_features=1024),
            ReLU(),
            BatchNorm1d(num_features=1024),
            Linear(in_features=1024, out_features=7 * 7 * 128),
            ReLU(),
            BatchNorm1d(num_features=7 * 7 * 128),
            Unflatten(channel=128, height=7, width=7),
            ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1),
            ReLU(),
            BatchNorm2d(num_features=64),
            ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1),
            Tanh(),
            Flatten(SAMPLE_SIZE))

    def forward(self, x):
        return self.sequential(x)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, Linear) or isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                xavier_uniform_(module.weight)


class MnistDcGan(Gan):
    def __init__(self, leaky_relu_negative_slope=0.01, device=torch.device('cpu')):
        super().__init__(device)
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

    def discriminator(self):
        return MnistDcGanDiscriminator(
            leaky_relu_negative_slope=self.leaky_relu_negative_slope)\
            .to(self.device)

    def generator(self):
        return MnistDcGanGenerator().to(self.device)

    @property
    def latent_vector_size(self):
        return LATENT_VECTOR_SIZE

    @property
    def sample_size(self):
        return SAMPLE_SIZE

    @property
    def image_size(self):
        return IMAGE_SIZE
