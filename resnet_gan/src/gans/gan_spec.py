import abc

from torch.nn import Module


class Gan:
    __metaclass__ = abc.ABC

    @property
    @abc.abstractmethod
    def latent_vector_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def image_size(self) -> int:
        pass

    @abc.abstractmethod
    def generator(self) -> Module:
        pass

    @abc.abstractmethod
    def discriminator(self) -> Module:
        pass