import torch
import abc


class GanModule(torch.nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def initialize(self):
        pass