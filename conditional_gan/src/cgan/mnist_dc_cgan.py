import torch
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, Linear, ReLU, ConvTranspose2d, BatchNorm2d, Tanh, MaxPool2d, \
    BatchNorm1d
from torch.nn.init import xavier_uniform_

from cgan.mnist_cgan import MnistCgan


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


class MnistDcGanDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = Sequential(
            Conv2d(in_channels=11,
                   out_channels=32,
                   kernel_size=5,
                   stride=1),
            LeakyReLU(negative_slope=0.01),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=32,
                   out_channels=64,
                   kernel_size=5,
                   stride=1),
            LeakyReLU(negative_slope=0.01),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(1024),
            Linear(in_features=1024, out_features=1024),
            LeakyReLU(negative_slope=0.01),
            Linear(in_features=1024, out_features=1))
        self.initialize()

    def forward(self, image_reshaped, label):
        n = image_reshaped.shape[0]
        image_reshaped = image_reshaped \
            .view(n, 1, MnistCgan.IMAGE_SIZE, MnistCgan.IMAGE_SIZE)
        label_reshaped = label \
            .view(n, 10, 1, 1) \
            .expand(n, 10, MnistCgan.IMAGE_SIZE, MnistCgan.IMAGE_SIZE)
        merged_image = torch.cat([image_reshaped, label_reshaped], dim=1)
        return self.sequential(merged_image)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, Linear) or isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                xavier_uniform_(module.weight)


class MnistDcGanGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = Sequential(
            Linear(in_features=MnistCgan.LATENT_VECTOR_SIZE + 10, out_features=1024),
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
            Flatten(MnistCgan.IMAGE_VECTOR_SIZE))
        self.initialize()

    def forward(self, latent_vector, label):
        zy = torch.cat([latent_vector, label], dim=1)
        return self.sequential(zy)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, Linear) or isinstance(module, Conv2d) or isinstance(module, ConvTranspose2d):
                xavier_uniform_(module.weight)


class MnistDcCgan(MnistCgan):
    def __init__(self, device=torch.device('cpu')):
        super().__init__(device)

    def discriminator(self):
        return MnistDcGanDiscriminator().to(self.device)

    def generator(self):
        return MnistDcGanGenerator().to(self.device)

    def discriminator_loss(self, batch_size=None):
        def loss(real_logit, fake_logit):
            real_prob = real_logit
            real_diff = real_prob - 1.0
            real_loss = real_diff.mul(real_diff).mean() / 2.0

            fake_prob = fake_logit
            fake_loss = fake_prob.mul(fake_prob).mean() / 2.0

            return real_loss + fake_loss

        return loss

    def generator_loss(self, batch_size=None):
        def loss(fake_logit):
            fake_prob = fake_logit
            fake_diff = fake_prob - 1.0
            return fake_diff.mul(fake_diff).mean() / 2.0

        return loss