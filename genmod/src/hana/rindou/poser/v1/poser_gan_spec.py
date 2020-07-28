import abc

from torch.nn import Module


class PoserGanSpec:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def requires_discriminator_optimization(self) -> bool:
        pass

    @abc.abstractmethod
    def image_size(self) -> int:
        pass

    @abc.abstractmethod
    def pose_size(self) -> int:
        pass

    @abc.abstractmethod
    def bone_parameter_count(self) -> int:
        pass

    @abc.abstractmethod
    def generator(self) -> Module:
        pass

    @abc.abstractmethod
    def discriminator(self) -> Module:
        pass
