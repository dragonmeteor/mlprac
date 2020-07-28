from torch.nn import Module, Sequential, Conv2d, InstanceNorm2d

from hana.bougain.nn.common import activation_module
from hana.rindou.nn2.init_function import create_init_function


class ResidualBlock(Module):
    def __init__(self, in_channels: int, one_pixel: bool = False, activation='relu'):
        super().__init__()
        init = create_init_function('he')
        if one_pixel:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = 1
        self.main = Sequential(
            init(Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)),
            InstanceNorm2d(in_channels, affine=True),
            activation_module(activation),
            init(Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)),
            InstanceNorm2d(in_channels, affine=True))

    def forward(self, x):
        return x + self.main(x)
