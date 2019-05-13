import torch

from torch.nn import Conv2d, ConvTranspose2d, Linear, Module, Sequential, LeakyReLU, AvgPool2d, Parameter
from torch.nn.init import calculate_gain, _calculate_correct_fan, normal_, kaiming_normal_, zeros_
import torch.nn.functional as F
import math

from gans.gan import GanModule, Gan
from gans.util import is_power2

LATENT_VECTOR_SIZE = 512


class PgGanConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nonlinearity='leaky_relu', nonlinearity_param=None):
        super().__init__()
        self.module = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        fan = _calculate_correct_fan(self.module.weight, 'fan_in')
        gain = calculate_gain(nonlinearity, nonlinearity_param)
        self.std = gain / math.sqrt(fan)

    def forward(self, input):
        #output = self.module(input)
        #return output * self.std
        return self.module(input)


class PgGanConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 nonlinearity='leaky_relu', nonlinearity_param=None):
        super().__init__()
        self.module = ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                      padding, output_padding, groups, bias, dilation)
        fan = _calculate_correct_fan(self.module.weight, 'fan_in')
        gain = calculate_gain(nonlinearity, nonlinearity_param)
        self.std = gain / math.sqrt(fan)

    def forward(self, input):
        #output = self.module(input) * self.std
        #return output
        return self.module(input)


class PgGanLinear(Module):
    def __init__(self, in_features, out_features, bias=True, nonlinearity='leaky_relu', nonlinearity_param=None):
        super().__init__()
        self.module = Linear(in_features, out_features, bias)
        fan = _calculate_correct_fan(self.module.weight, 'fan_in')
        gain = calculate_gain(nonlinearity, nonlinearity_param)
        self.std = gain / math.sqrt(fan)

    def forward(self, input):
        output = self.module(input) * self.std
        #return output
        #return output
        return self.module(input)


class PixelWiseNorm(Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor):
        #return input / ((input ** 2).mean(dim=1, keepdim=True) + self.epsilon).sqrt()
        return input


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
        input_channels = CHANNEL_COUNT_BY_SIZE[block_size]


def to_rgb_layer(size: int):
    return PgGanConv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size],
                       out_channels=3,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       nonlinearity='linear')


def initialize_modules(gan_module: Module):
    for module in gan_module.modules():
        if False:
            if isinstance(module, PgGanLinear):
                normal_(module.module.weight)
                zeros_(module.module.bias)
            if isinstance(module, PgGanConv2d):
                normal_(module.module.weight)
                zeros_(module.module.bias)
            if isinstance(module, PgGanConvTranspose2d):
                normal_(module.module.weight)
                zeros_(module.module.bias)


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
        initialize_modules(self)


class PgGanGeneratorTransition(GanModule):
    def __init__(self, size):
        super().__init__()
        assert size >= 4
        assert is_power2(size)
        self.size = size
        self.alpha = 0.0
        create_generator_blocks(self, size)
        self.to_rgb_layers = [
            to_rgb_layer(size // 2),
            to_rgb_layer(size)
        ]
        self.add_module("to_rgb_%05d" % (self.size // 2), self.to_rgb_layers[0])
        self.add_module("to_rgb_%05d" % (self.size), self.to_rgb_layers[1])

    def forward(self, input):
        value = input
        for block in self.blocks[:-1]:
            value = block(value)
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
        PixelWiseNorm(),
        PgGanConv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size * 2],
                    out_channels=CHANNEL_COUNT_BY_SIZE[size],
                    kernel_size=3,
                    padding=1,
                    nonlinearity='leaky_relu',
                    nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm(),
        AvgPool2d(kernel_size=2, stride=2))


def discriminator_score_block():
    return Sequential(
        #MiniBatchStddev(),
        #PgGanConv2d(in_channels=513, out_channels=512, kernel_size=3, padding=1,
        #            nonlinearity='leaky_relu', nonlinearity_param=0.2),
        PgGanConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1,
                   nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm(),
        PgGanConv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0,
                    nonlinearity='leaky_relu', nonlinearity_param=0.2),
        LeakyReLU(negative_slope=0.2),
        PixelWiseNorm(),
        Flatten(512),
        PgGanLinear(in_features=512, out_features=1, nonlinearity='linear'))


def create_discriminator_blocks(gan_module: Module, size: int):
    block_size = size
    while block_size > 4:
        add_block(gan_module, "block_%05d" % block_size, discriminator_block(block_size // 2))
        block_size //= 2
    add_block(gan_module, "score", discriminator_score_block())


class PgGanDiscriminator(GanModule):
    def __init__(self, size):
        super().__init__()
        assert size >= 4
        assert is_power2(size)
        self.size = size
        self.blocks = []
        add_block(self, "from_rgb_%05d" % size, from_rgb_block(size))
        create_discriminator_blocks(self, size)

    def forward(self, input):
        value = input
        for block in self.blocks:
            value = block(value)
        return value

    def initialize(self):
        initialize_modules(self)


class PgGanDiscriminatorTransition(GanModule):
    def __init__(self, size):
        super().__init__()
        assert size >= 4
        assert is_power2(size)
        self.size = size
        self.alpha = 0.0
        self.blocks = []

        self.from_rgb_blocks = [
            from_rgb_block(size // 2),
            from_rgb_block(size)
        ]
        self.add_module("from_rgb_%05d" % (size // 2), self.from_rgb_blocks[0])
        self.add_module("from_rgb_%05d" % size, self.from_rgb_blocks[1])

        create_discriminator_blocks(self, size)

    def forward(self, input):
        input_rgb = self.from_rgb_blocks[1](input)
        new_value = self.blocks[0](input_rgb)
        old_value = self.from_rgb_blocks[0](F.avg_pool2d(input, kernel_size=2, stride=2))
        value = (1.0 - self.alpha) * old_value + self.alpha * new_value
        for block in self.blocks[1:]:
            value = block(value)
        return value

    def initialize(self):
        initialize_modules(self)


class PgGan(Gan):
    def __init__(self, size: int, device=torch.device('cpu')):
        super().__init__(device)
        assert size >= 4
        assert is_power2(size)
        self.size = size

    @property
    def sample_size(self) -> int:
        return self.size * self.size

    @property
    def latent_vector_size(self) -> int:
        return LATENT_VECTOR_SIZE

    @property
    def image_size(self) -> int:
        return self.size

    def discriminator(self) -> GanModule:
        return PgGanDiscriminator(self.size).to(self.device)

    def generator(self) -> GanModule:
        return PgGanGenerator(self.size).to(self.device)


class PgGanTransition(Gan):
    def __init__(self, size: int, device=torch.device('cpu')):
        super().__init__(device)
        assert size >= 4
        assert is_power2(size)
        self.size = size

    @property
    def sample_size(self) -> int:
        return self.size * self.size

    @property
    def latent_vector_size(self) -> int:
        return LATENT_VECTOR_SIZE

    @property
    def image_size(self) -> int:
        return self.size

    def discriminator(self) -> GanModule:
        return PgGanDiscriminatorTransition(self.size).to(self.device)

    def generator(self) -> GanModule:
        return PgGanGeneratorTransition(self.size).to(self.device)


if __name__ == "__main__":
    size = 4
    while size <= 64:
        if size > 4:
            D = PgGanDiscriminatorTransition(size)
            print("DiscriminatorTransition(%d)" % size)
            for name in D.state_dict().keys():
                print(name)
            print()

        D = PgGanDiscriminator(size)
        print("Discriminator(%d)" % size)
        for name in D.state_dict().keys():
            print(name)
        print()

        size *= 2

    size = 4
    while size <= 64:
        if size > 4:
            G = PgGanGeneratorTransition(size)
            print("GeneratorTransition(%d)" % size)
            for name in G.state_dict().keys():
                print(name)
            print()

        G = PgGanGenerator(size)
        print("Generator(%d)" % size)
        for name in G.state_dict().keys():
            print(name)
        print()

        size *= 2


    #print(DT16(torch.Tensor(4,3,16,16)).shape)


    #G4 = PgGanGenerator(4)
    # for key,value in G4.state_dict().items():
    #    print(key)
    #    print(value.shape)
    #print(G4.blocks[1].weight.shape)

    #GT8 = PgGanGeneratorTransition(8)
    #GT8.initialize()
    #GT8.load_state_dict(G4.state_dict(), strict=False)

    # print(G4.blocks[0][1].weight - GT8.blocks[0][1].weight)
    # print(len(G4.blocks))

    # print(GT8.blocks[1][1].weight)
    # print(len(GT8.blocks))

    #G8 = PgGanGenerator(8)

    #G8.load_state_dict(GT8.state_dict(), strict=False)
    # print(G(torch.zeros(16, 512)).shape)

    # GT = PgGanGeneratorTransition(8)
    # print(GT(torch.zeros(16, 512)).shape)

    # D = PgGanDiscriminator(16)
    # print(D(torch.zeros(8, 3, 16, 16)).shape)

    # DT = PgGanDiscriminatorTransition(16)
    # print(D(torch.zeros(8, 3, 16, 16)).shape)
