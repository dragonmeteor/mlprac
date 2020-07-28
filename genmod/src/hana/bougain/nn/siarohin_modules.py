from typing import List

import torch
from torch.nn import Module, Conv2d, Sequential, AvgPool2d, InstanceNorm2d, ModuleList
from torch.nn.functional import interpolate

from hana.bougain.nn.common import activation_module
from hana.rindou.nn2.init_function import create_init_function


class UpsampleBy2(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return interpolate(x, scale_factor=2)


class SiarohinDownsampleBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation: str = 'relu'):
        super().__init__()
        init = create_init_function('he')
        self.body = Sequential(
            init(Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False)),
            InstanceNorm2d(num_features=out_channels, affine=True),
            activation_module(activation),
            AvgPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        return self.body(x)


class SiarohinUpsampleBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation: str = 'relu'):
        super().__init__()
        init = create_init_function('he')
        self.body = Sequential(
            UpsampleBy2(),
            init(Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False)),
            InstanceNorm2d(num_features=out_channels, affine=True),
            activation_module(activation))

    def forward(self, x):
        return self.body(x)


class SiarohinUNetEncoder(Module):
    def __init__(self, in_channels: int, num_blocks: int, max_channels=1024, activation: str = 'relu'):
        super().__init__()
        downsample_blocks = []
        num_channels = in_channels
        self.num_output_channels = [in_channels]
        for block_index in range(num_blocks):
            downsample_blocks.append(SiarohinDownsampleBlock(
                in_channels=min(max_channels, num_channels),
                out_channels=min(max_channels, num_channels * 2),
                activation=activation))
            self.num_output_channels.append(min(max_channels, num_channels * 2))
            num_channels = num_channels * 2
        self.blocks = ModuleList(downsample_blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        output = [x]
        for block in self.blocks:
            x = block(x)
            output.append(x)
        return output

    def get_num_outout_channels(self):
        return self.num_output_channels


class SiarohinSameSizeBlock(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 activation: str = 'relu'):
        super().__init__()
        init = create_init_function('he')
        self.body = Sequential(
            init(Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False)),
            InstanceNorm2d(num_features=out_channels, affine=True),
            activation_module(activation))

    def forward(self, x):
        return self.body(x)


class SiarohinUNetDecoder(Module):
    def __init__(self,
                 out_channels: int,
                 num_blocks: int,
                 num_skipover_channels: List[int],
                 max_channels=1024, activation: str = 'relu'):
        super().__init__()
        assert len(num_skipover_channels) == num_blocks + 1
        self.num_blocks = num_blocks
        in_channels = out_channels * (2 ** num_blocks)
        upsample_blocks = []
        num_skipover_channels = num_skipover_channels[:]
        num_skipover_channels.reverse()
        self.num_output_channels = [num_skipover_channels[0]]

        num_channels = in_channels
        upsample_blocks.append(
            SiarohinUpsampleBlock(
                in_channels=min(max_channels, num_channels),
                out_channels=min(max_channels, num_channels // 2),
                activation=activation))
        num_channels //= 2
        self.num_output_channels.append(num_channels)

        for block_index in range(1, num_blocks):
            upsample_blocks.append(
                SiarohinUpsampleBlock(
                    in_channels=min(max_channels, num_channels) + num_skipover_channels[block_index],
                    out_channels=min(max_channels, num_channels // 2),
                    activation=activation))
            self.num_output_channels.append(min(max_channels, num_channels // 2))
            num_channels //= 2

        upsample_blocks.append(SiarohinSameSizeBlock(
            in_channels=min(max_channels, num_channels) + num_skipover_channels[-1],
            out_channels=min(max_channels, num_channels),
            activation=activation))
        self.num_output_channels[-1] = min(max_channels, num_channels)


        self.blocks = ModuleList(upsample_blocks)
        self.num_output_channels.reverse()

    def forward(self, encoder_output: List[torch.Tensor]) -> List[torch.Tensor]:
        output = [encoder_output[-1]]
        y = self.blocks[0](encoder_output[-1])
        output.append(y)
        for i in range(1, self.num_blocks):
            x = torch.cat([encoder_output[-i - 1], y], dim=1)
            y = self.blocks[i](x)
            output.append(y)
        x = torch.cat([encoder_output[0], y], dim=1)
        y = self.blocks[-1](x)
        output[-1] = (y)
        output.reverse()
        return output

    def get_num_output_channels(self):
        return self.num_output_channels


class SiarohinUNet(Module):
    def __init__(self, in_channels: int, num_blocks: int, max_channels=1024, activation: str = 'relu'):
        super().__init__()
        self.encoder = SiarohinUNetEncoder(
            in_channels=in_channels,
            num_blocks=num_blocks,
            max_channels=max_channels,
            activation=activation)
        self.decoder = SiarohinUNetDecoder(
            out_channels=in_channels,
            num_blocks=num_blocks,
            num_skipover_channels=self.encoder.get_num_outout_channels(),
            max_channels=max_channels,
            activation=activation)

    def forward(self, x):
        return self.decoder(self.encoder(x))[0]


if __name__ == "__main__":
    cuda = torch.device('cuda')
    x = torch.zeros(4, 32, 256, 256, device=cuda)
    encoder = SiarohinUNetEncoder(in_channels=32, num_blocks=5, max_channels=1024).to(cuda)
    print(encoder.get_num_outout_channels())
    decoder = SiarohinUNetDecoder(out_channels=32, num_blocks=5,
                                  num_skipover_channels=encoder.get_num_outout_channels(),
                                  max_channels=1024).to(cuda)
    output = encoder(x)
    y = decoder(output)
    print(decoder.get_num_output_channels())
    print([x.shape for x in y])
