import torch

from torch.nn import Conv2d, ConvTranspose2d, Linear, Module, Sequential, LeakyReLU, AvgPool2d, Upsample
from torch.nn.init import calculate_gain, _calculate_correct_fan, normal_
import torch.nn.functional as F
import math

from gans.gan import GanModule

LATENT_VECTOR_SIZE = 512


class PgGanConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nonlinearity='leaky_relu', nonlinearity_param=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        fan = _calculate_correct_fan(self.weight, 'fan_in')
        gain = calculate_gain(nonlinearity, nonlinearity_param)
        self.std = gain / math.sqrt(fan)

    def forward(self, input):
        return F.conv2d(input, self.weight * self.std, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class PgGanConvTranspose2d(ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 nonlinearity='leaky_relu', nonlinearity_param=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, output_padding, groups, bias, dilation)
        fan = _calculate_correct_fan(self.weight, 'fan_in')
        gain = calculate_gain(nonlinearity, nonlinearity_param)
        self.std = gain / math.sqrt(fan)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(
            input, self.weight * self.std, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class PgGanLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, nonlinearity='leaky_relu', nonlinearity_param=None):
        super().__init__(in_features, out_features, bias)
        fan = _calculate_correct_fan(self.weight, 'fan_in')
        gain = calculate_gain(nonlinearity, nonlinearity_param)
        self.std = gain / math.sqrt(fan)

    def forward(self, input):
        return F.linear(input, self.weight * self.std, self.bias)


class PixelWiseNorm(Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor):
        return input * (input.mul(input).sum(dim=1, keepdim=True) + self.epsilon).rsqrt()


class MiniBatchStddev(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        std = input.std(dim=0, keepdim=True).mean().item()
        std_feature = torch.ones(input.shape[0], 1, input.shape[2], input.shape[3], device=input.device) * std
        return torch.cat((input, std_feature), dim=1)


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


class DoubleSize(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = F.interpolate(input, scale_factor=2, mode='nearest')
        return output


def is_power2(x):
    return x != 0 and ((x & (x - 1)) == 0)


CHANNEL_COUNT_BY_SIZE = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
    1024: 16
}


def generator_first_block():
    return Sequential(
        Unflatten(LATENT_VECTOR_SIZE, 1, 1),
        PgGanConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4,
                             nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm(),
        PgGanConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1,
                    nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm())


def generator_block(block_size, in_channels):
    out_channels = CHANNEL_COUNT_BY_SIZE[block_size]
    return Sequential(
        DoubleSize(),
        PgGanConv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm(),
        PgGanConv2d(in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm())


def add_block(gan_module: Module, name: str, block: Module):
    gan_module.add_module(name, block)
    gan_module.blocks.append(block)


def create_generator_blocks(gan_module: Module, size):
    gan_module.blocks = []

    add_block(gan_module, "block_00004", generator_first_block())

    block_size = 4
    input_channels = 512
    while block_size < size:
        block_size *= 2
        add_block(gan_module,
                  "block_%05d" % block_size,
                  generator_block(block_size, input_channels))
        input_channels = CHANNEL_COUNT_BY_SIZE[size]


def to_rgb_layer(size: int):
    return PgGanConv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size],
                       out_channels=3,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       nonlinearity='linear')


def initialize_modules(gan_module: Module):
    for module in gan_module.modules():
        if isinstance(module, PgGanConv2d) \
                or isinstance(module, PgGanConvTranspose2d) \
                or isinstance(module, PgGanLinear):
            normal_(module.weight)


class PgGanGenerator(GanModule):
    def __init__(self, size):
        super().__init__()
        assert size >= 4
        assert is_power2(size)
        self.size = size
        create_generator_blocks(self, size)
        add_block(self, "to_rgb_%05d" % self.size, to_rgb_layer(size))

    def forward(self, input):
        value = input
        for block in self.blocks:
            value = block(value)
        return value

    def initialize(self):
        for module in self.modules():
            if isinstance(module, PgGanConv2d) or isinstance(module, PgGanConvTranspose2d):
                normal_(module.weight)


class PgGanGeneratorTransition(GanModule):
    def __init__(self, size):
        super().__init__()
        assert size >= 4
        assert is_power2(size)
        self.size = size
        self.alpha = 0.0
        create_generator_blocks(self, size)
        self.to_rgb_layers = [
            to_rgb_layer(size),
            to_rgb_layer(size)
        ]
        self.add_module("to_rgb_%05d" % (self.size // 2), self.to_rgb_layers[0])
        self.add_module("to_rgb_%05d" % (self.size), self.to_rgb_layers[1])

    def forward(self, input):
        value = input
        for block in self.blocks[:-1]:
            value = block(input)
        before_last = self.to_rgb_layers[0](F.interpolate(value, scale_factor=2))
        last = self.to_rgb_layers[1](self.blocks[-1](value))
        return (1.0 - self.alpha) * before_last + self.alpha * last

    def initialize(self):
        initialize_modules(self)


def from_rgb_block(size: int):
    return Sequential(
        PgGanConv2d(in_channels=3,
                    out_channels=CHANNEL_COUNT_BY_SIZE[size],
                    kernel_size=1,
                    nonlinearity='leaky_relu',
                    nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm())


def discriminator_block(size: int):
    return Sequential(
        PgGanConv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size * 2],
                    out_channels=CHANNEL_COUNT_BY_SIZE[size * 2],
                    kernel_size=3,
                    padding=1,
                    nonlinearity='leaky_relu',
                    nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PgGanConv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size * 2],
                    out_channels=CHANNEL_COUNT_BY_SIZE[size],
                    kernel_size=3,
                    padding=1,
                    nonlinearity='leaky_relu',
                    nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        AvgPool2d(kernel_size=2, stride=2))


def discriminator_score_block():
    return Sequential(
        MiniBatchStddev(),
        PgGanConv2d(in_channels=513, out_channels=512, kernel_size=3, padding=1,
                    nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PgGanConv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0,
                    nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        Flatten(512),
        PgGanLinear(in_features=512, out_features=1, nonlinearity='linear'))


class PgGanDiscriminator(GanModule):
    def __init__(self, size):
        super().__init__()
        assert size >= 4
        assert is_power2(size)
        self.size = size

        add_block(self, "from_rgb_%05d" % size, from_rgb_block(size))

        block_size = size
        while block_size > 4:
            add_block(self, "block_%05d" % size, discriminator_block(block_size // 2))
            block_size //= 2

        add_block(self, "score", discriminator_score_block())

    def forward(self, input):
        value = input
        for block in self.blocks:
            value = block(value)
        return value

    def initialize(self):
        initialize_modules(self)


if __name__ == "__main__":
    G = PgGanGenerator(8)
    print(G(torch.zeros(16, 512)).shape)

    GT = PgGanGeneratorTransition(8)
    GT(torch.zeros(16, 512))
