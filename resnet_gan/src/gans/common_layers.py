import torch
from torch.nn import Module, functional as F


class DoubleSize(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = F.interpolate(input, scale_factor=2, mode='nearest')
        return output


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