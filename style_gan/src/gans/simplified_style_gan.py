from typing import Dict, List

import torch

from torch.nn import Module, Sequential, Linear, LeakyReLU, Conv2d, Parameter
from torch.nn import functional as F
from torch.nn.init import ones_, zeros_

from gans.gan_module import GanModule
from gans.simplified_pggan import PgGanDiscriminator, PgGanDiscriminatorTransition
from gans.style_gan_spec import StyleGan

LATENT_VECTOR_SIZE = 512
LEAKY_RELU_NEGATIVE_SLOPE = 0.2

NUM_CHANNELS_BY_IMAGE_SIZE = {
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


class MappingModule(GanModule):
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


def create_noise(input_image: torch.Tensor, noise_factor: torch.Tensor, input_noise: torch.Tensor = None):
    n = input_image.shape[0]
    c = input_image.shape[1]
    h = input_image.shape[2]
    w = input_image.shape[3]

    if input_noise is None:
        raw_noise = torch.randn(n, 1, h, w, device=input_image.device).expand(n, c, h, w)
    else:
        raw_noise = input_noise.view(1, 1, h, w).expand(n, c, h, w)
    return raw_noise * noise_factor


class GeneratorBlock(GanModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
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
                latent_vector: torch.Tensor,
                noise_1: torch.Tensor = None,
                noise_2: torch.Tensor = None):
        upsampled_image = self.upsample(input_image)
        conv_1_image = self.convolve_1(upsampled_image)
        noise_1_image = conv_1_image + create_noise(conv_1_image, self.noise_1_factor, noise_1)
        adain_1_image = AdaIN(noise_1_image, self.weight_to_new_mean_1(latent_vector),
                              self.weight_to_new_std_1(latent_vector))
        conv_2_image = self.convolve_2(adain_1_image)
        noise_2_image = conv_2_image + create_noise(conv_2_image, self.noise_2_factor, noise_2)
        adain_2_image = AdaIN(noise_2_image, self.weight_to_new_mean_2(latent_vector),
                              self.weight_to_new_std_2(latent_vector))
        return adain_2_image

    def convolve_1(self, input_image: torch.Tensor):
        return F.leaky_relu(self.conv_1(input_image), negative_slope=self.leaky_relu_negative_slope)

    def convolve_2(self, input_image: torch.Tensor):
        return F.leaky_relu(self.conv_2(input_image), negative_slope=self.leaky_relu_negative_slope)

    def initialize(self):
        pass


class GeneratorFirstBlock(GanModule):
    def __init__(self,
                 image_size: int = 4,
                 out_channels: int = 512,
                 latent_vector_size: int = LATENT_VECTOR_SIZE,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()
        self.image_size = image_size
        self.out_channels = out_channels
        self.latent_vector_size = latent_vector_size
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

        self.start_image = Parameter(torch.Tensor(1, out_channels, image_size, image_size))
        ones_(self.start_image)

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

    def forward(self,
                latent_vector: torch.Tensor,
                noise_1: torch.Tensor = None,
                noise_2: torch.Tensor = None):
        n = latent_vector.shape[0]
        start_image = self.start_image.expand(n, self.out_channels, self.image_size, self.image_size)
        noise_1_image = start_image + create_noise(start_image, self.noise_1_factor, noise_1)
        adain_1_image = AdaIN(noise_1_image, self.weight_to_new_mean_1(latent_vector),
                              self.weight_to_new_std_1(latent_vector))
        conv_2_image = self.convolve_2(adain_1_image)
        noise_2_image = conv_2_image + create_noise(conv_2_image, self.noise_2_factor, noise_2)
        adain_2_image = AdaIN(noise_2_image, self.weight_to_new_mean_2(latent_vector),
                              self.weight_to_new_std_2(latent_vector))
        return adain_2_image

    def convolve_2(self, input_image: torch.Tensor):
        return F.leaky_relu(self.conv_2(input_image), negative_slope=self.leaky_relu_negative_slope)

    def initialize(self):
        pass


def set_generator_properites(network,
                             image_size: int,
                             latent_vector_size: int = LATENT_VECTOR_SIZE,
                             leaky_relu_negative_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
                             num_channels_by_image_size: Dict[int, int] = None):
    network.image_size = image_size
    network.latent_vector_size = latent_vector_size
    network.leaky_relu_negative_slope = leaky_relu_negative_slope
    if num_channels_by_image_size is None:
        num_channels_by_image_size = NUM_CHANNELS_BY_IMAGE_SIZE
    network.num_channels_by_image_size = num_channels_by_image_size


def add_first_generator_block(network: Module):
    first_block = GeneratorFirstBlock(image_size=4,
                                      out_channels=network.num_channels_by_image_size[4],
                                      latent_vector_size=network.latent_vector_size,
                                      leaky_relu_negative_slope=network.leaky_relu_negative_slope)
    network.add_module("block_%05d" % 4, first_block)


def add_generator_blocks(network: Module):
    network.blocks = []
    size = 8
    while size <= network.image_size:
        block = GeneratorBlock(in_channels=network.num_channels_by_image_size[size // 2],
                               out_channels=network.num_channels_by_image_size[size],
                               latent_vector_size=network.latent_vector_size,
                               leaky_relu_negative_slope=network.leaky_relu_negative_slope)
        network.add_module("block_%05d" % size, block)
        network.blocks.append(block)
        size *= 2


def to_rgb_layer(size: int, num_channels_by_image_size: Dict[int, int] = None):
    if num_channels_by_image_size is None:
        num_channels_by_image_size = NUM_CHANNELS_BY_IMAGE_SIZE
    return Conv2d(in_channels=num_channels_by_image_size[size],
                  out_channels=3,
                  kernel_size=1,
                  stride=1,
                  padding=0)


def call_first_generator_block(block: Module,
                               latent_vector: torch.Tensor,
                               noise_image: List[List[torch.Tensor]]):
    if noise_image is not None:
        return block(latent_vector, noise_image[0][0], noise_image[0][1])
    else:
        return block(latent_vector, None, None)


def call_generator_block(block: Module,
                         input_image: torch.Tensor,
                         latent_vector: torch.Tensor,
                         noise_image_index: int,
                         noise_image: List[List[torch.Tensor]]):
    if noise_image is not None:
        return block(input_image,
                     latent_vector,
                     noise_image[noise_image_index][0],
                     noise_image[noise_image_index][1])
    else:
        return block(input_image, latent_vector, None, None)


class GeneratorModule(GanModule):
    def __init__(self,
                 image_size: int,
                 latent_vector_size: int = LATENT_VECTOR_SIZE,
                 leaky_relu_negative_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
                 num_channels_by_image_size: Dict[int, int] = None):
        super().__init__()
        set_generator_properites(self,
                                 image_size,
                                 latent_vector_size,
                                 leaky_relu_negative_slope,
                                 num_channels_by_image_size)
        add_first_generator_block(self)
        add_generator_blocks(self)

        rgb_layer = to_rgb_layer(size=self.image_size, num_channels_by_image_size=self.num_channels_by_image_size)
        self.add_module("to_rgb_%05d" % self.image_size, rgb_layer)
        self.to_rgb_layers = [rgb_layer]

    def forward(self, latent_vector: torch.Tensor, noise_image: List[List[torch.Tensor]] = None):
        if noise_image is not None:
            assert len(noise_image) >= len(self.blocks) + 1
            for item in noise_image:
                assert len(item) >= 2

        value = call_first_generator_block(self.block_00004, latent_vector, noise_image)
        noise_image_index = 1
        for block in self.blocks:
            value = call_generator_block(block, value, latent_vector, noise_image_index, noise_image)
            noise_image_index += 1

        return self.to_rgb_layers[0](value)

    def initialize(self):
        pass


class GeneratorTransitionModule(GanModule):
    def __init__(self,
                 image_size: int,
                 latent_vector_size: int = LATENT_VECTOR_SIZE,
                 leaky_relu_negative_slope: float = LEAKY_RELU_NEGATIVE_SLOPE,
                 num_channels_by_image_size: Dict[int, int] = None):
        super().__init__()
        set_generator_properites(self,
                                 image_size,
                                 latent_vector_size,
                                 leaky_relu_negative_slope,
                                 num_channels_by_image_size)
        add_first_generator_block(self)
        add_generator_blocks(self)

        self.to_rgb_layers = [
            to_rgb_layer(self.image_size // 2, self.num_channels_by_image_size),
            to_rgb_layer(self.image_size, self.num_channels_by_image_size)
        ]
        self.add_module("to_rgb_%05d" % (self.image_size // 2), self.to_rgb_layers[0])
        self.add_module("to_rgb_%05d" % self.image_size, self.to_rgb_layers[1])

        self.alpha = 0.0

    def forward(self, latent_vector: torch.Tensor, noise_image: List[List[torch.Tensor]] = None):
        if noise_image is not None:
            assert len(noise_image) >= len(self.blocks) + 1
            for item in noise_image:
                assert len(item) >= 2

        value = call_first_generator_block(self.block_00004, latent_vector, noise_image)
        noise_image_index = 1
        for block in self.blocks[:-1]:
            value = call_generator_block(block=block,
                                         input_image=value,
                                         latent_vector=latent_vector,
                                         noise_image_index=noise_image_index,
                                         noise_image=noise_image)
            noise_image_index += 1

        lowres_image = self.to_rgb_layers[0](value)
        lowres_image = F.interpolate(lowres_image, scale_factor=2, mode='nearest')

        highres_image = self.to_rgb_layers[1](
            call_generator_block(block=self.blocks[-1],
                                 input_image=value,
                                 latent_vector=latent_vector,
                                 noise_image_index=len(self.blocks),
                                 noise_image=noise_image))

        return lowres_image * (1.0 - self.alpha) + highres_image * self.alpha

    def initialize(self):
        pass


class SimplifiedStyleGan(StyleGan):
    def __init__(self,
                 leaky_relu_negative_slope=LEAKY_RELU_NEGATIVE_SLOPE,
                 num_channels_by_image_size: Dict[int, int] = None):
        super().__init__()
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        if num_channels_by_image_size is None:
            num_channels_by_image_size = NUM_CHANNELS_BY_IMAGE_SIZE
        self.num_channels_by_image_size = num_channels_by_image_size

    @property
    def latent_vector_size(self) -> int:
        return LATENT_VECTOR_SIZE

    def mapping_module(self) -> GanModule:
        return MappingModule(self.latent_vector_size, self.leaky_relu_negative_slope)

    def generator_module_stabilize(self, image_size) -> GanModule:
        return GeneratorModule(image_size,
                               self.latent_vector_size,
                               self.leaky_relu_negative_slope,
                               self.num_channels_by_image_size)

    def discriminator_stabilize(self, image_size) -> GanModule:
        return PgGanDiscriminator(image_size)

    def generator_module_transition(self, image_size) -> GanModule:
        return GeneratorTransitionModule(image_size,
                                         self.latent_vector_size,
                                         self.leaky_relu_negative_slope,
                                         self.num_channels_by_image_size)

    def discriminator_transition(self, image_size) -> GanModule:
        return PgGanDiscriminatorTransition(image_size)
