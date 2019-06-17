import torch

from torch.nn import BatchNorm2d
import torch.nn.functional as F
from torch.nn import Conv2d, Module, AvgPool2d
from torch.nn.init import kaiming_normal_

from gans.common_layers import DoubleSize
from gans.spectral_norm_layers import SnConv2d


class ResidualBlock2d(Module):
    def __init__(self,
                 in_channels, out_channels,
                 resample=None, use_batchnorm=False, use_last_relu=True, use_spectral_normalization=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample
        self.use_batchnorm = use_batchnorm
        self.use_last_relu = use_last_relu
        self.use_spectral_normalization = use_spectral_normalization

        if resample == "up":
            self.shortcut = DoubleSize()
        elif resample == "down":
            self.shortcut = AvgPool2d(kernel_size=2, stride=2)
        else:
            self.shortcut = None

        if use_spectral_normalization:
            self.conv1 = SnConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        kaiming_normal_(self.conv1.weight)

        if use_spectral_normalization:
            self.conv2 = SnConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        else:
            self.conv2 = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        kaiming_normal_(self.conv2.weight)

        if use_batchnorm:
            self.batchnorm1 = BatchNorm2d(num_features=in_channels)
            self.batchnorm2 = BatchNorm2d(num_features=in_channels)
        else:
            self.batchnorm1 = None
            self.batchnorm2 = None

        if in_channels != out_channels:
            if use_spectral_normalization:
                self.channel_match = SnConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            else:
                self.channel_match = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            kaiming_normal_(self.channel_match.weight)
        else:
            self.channel_match = None

    def forward(self, input):
        current = input

        current = self.conv1(current)

        if self.batchnorm1 is not None:
            current = self.batchnorm1(current)

        current = F.relu(current)

        if self.resample == "up":
            current = F.interpolate(current, scale_factor=2, mode='nearest')

        current = self.conv2(current)

        if self.batchnorm2 is not None:
            current = self.batchnorm2(current)

        if self.resample == "down":
            current = F.avg_pool2d(current, 2, stride=2)

        if self.shortcut is None:
            current = current + input
        else:
            current = current + self.shortcut(input)

        if self.use_last_relu:
            current = F.relu(current)

        if self.channel_match is not None:
            current = self.channel_match(current)

        return current


if __name__ == "__main__":
    block = ResidualBlock2d(in_channels=8, out_channels=16, resample="down", use_batchnorm=True, use_last_relu=True)
    output = block(torch.zeros(4, 8, 16, 16))
    print(output.shape)

    print(block.state_dict().keys())
