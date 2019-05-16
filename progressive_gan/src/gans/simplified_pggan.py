import torch
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, Linear, Module, Sequential, LeakyReLU, AvgPool2d

from gans.gan_module import GanModule, Gan
from gans.pggan_spec import PgGan
from gans.util import is_power2

LATENT_VECTOR_SIZE = 512


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
        ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4),
        LeakyReLU(negative_slope=0.2),
        Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        LeakyReLU(negative_slope=0.2))


def generator_block(block_size, in_channels):
    out_channels = CHANNEL_COUNT_BY_SIZE[block_size]
    return Sequential(
        DoubleSize(),
        Conv2d(in_channels=in_channels,
               out_channels=out_channels,
               kernel_size=3,
               padding=1),
        LeakyReLU(negative_slope=0.2),
        Conv2d(in_channels=out_channels,
               out_channels=out_channels,
               kernel_size=3,
               padding=1),
        LeakyReLU(negative_slope=0.2))


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
    return Conv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size],
                  out_channels=3,
                  kernel_size=1,
                  stride=1,
                  padding=0)


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
        pass


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
        pass


def from_rgb_block(size: int):
    return Sequential(
        Conv2d(in_channels=3,
               out_channels=CHANNEL_COUNT_BY_SIZE[size],
               kernel_size=1),
        LeakyReLU(negative_slope=0.2))


def discriminator_block(size: int):
    return Sequential(
        Conv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size * 2],
               out_channels=CHANNEL_COUNT_BY_SIZE[size * 2],
               kernel_size=3,
               padding=1),
        LeakyReLU(negative_slope=0.2),
        Conv2d(in_channels=CHANNEL_COUNT_BY_SIZE[size * 2],
               out_channels=CHANNEL_COUNT_BY_SIZE[size],
               kernel_size=3,
               padding=1),
        LeakyReLU(negative_slope=0.2),
        AvgPool2d(kernel_size=2, stride=2))


def discriminator_score_block():
    return Sequential(
        Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        LeakyReLU(negative_slope=0.2),
        Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0),
        LeakyReLU(negative_slope=0.2),
        Flatten(512),
        Linear(in_features=512, out_features=1))


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
        pass


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
        pass


class SimplifiedPgGan(PgGan):
    def __init__(self):
        super().__init__(self)

    @property
    def latent_vector_size(self) -> int:
        return 512

    def generator_stabilize(self, image_size: int) -> GanModule:
        return PgGanGenerator(image_size)

    def generator_transition(self, image_size: int) -> GanModule:
        return PgGanGeneratorTransition(image_size)

    def discriminator_stabilize(self, image_size: int) -> GanModule:
        return PgGanDiscriminator(image_size)

    def discriminator_transition(self, image_size: int) -> GanModule:
        return PgGanDiscriminatorTransition(image_size)


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
