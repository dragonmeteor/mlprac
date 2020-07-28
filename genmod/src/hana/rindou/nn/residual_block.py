from torch.nn import Module, Sequential, Conv2d, InstanceNorm2d, ReLU, BatchNorm2d
import torch.nn.functional as F

from hana.rindou.nn.init_function import create_init_function


# Copied from https://github.com/albertpumarola/GANimation/blob/master/networks/generator_wasserstein_gan.py
class ResidualBlock(Module):
    def __init__(self, dim, initialization_method='he', one_pixel=False):
        super().__init__()
        init = create_init_function(initialization_method)
        if one_pixel:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = 1
        self.main = Sequential(
            init(Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False)),
            InstanceNorm2d(dim, affine=True),
            ReLU(),
            init(Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False)),
            InstanceNorm2d(dim, affine=True))

    def forward(self, x):
        return x + self.main(x)


class ResidualBlock2d(Module):
    def __init__(self,
                 num_channel: int,
                 normalization_mode: str = 'none',
                 use_last_relu=False,
                 initialization_method="he",
                 one_pixel: bool = False):
        super().__init__()
        init = create_init_function(initialization_method)
        self.use_last_relu = use_last_relu

        if one_pixel:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = 1
        self.conv1 = init(
            Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernel_size, padding=padding))
        self.conv2 = init(
            Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=kernel_size, padding=padding))

        if normalization_mode == 'batch':
            self.norm1 = BatchNorm2d(num_features=num_channel)
            self.norm2 = BatchNorm2d(num_features=num_channel)
        elif normalization_mode == 'instance':
            self.norm1 = InstanceNorm2d(num_features=num_channel)
            self.norm2 = InstanceNorm2d(num_features=num_channel)
        else:
            self.norm1 = lambda x: x
            self.norm2 = lambda x: x

    def forward(self, input):
        current = self.conv1(input)
        current = self.norm1(current)
        current = F.relu(current)
        current = self.conv2(current)
        current = self.norm2(current)
        current = current + input
        if self.use_last_relu:
            current = F.relu(current)
        return current
