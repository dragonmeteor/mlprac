import abc

from torch.nn import Module


class PoseRegressorSpec:
    __metaclass__ = abc.ABCMeta

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
    def regressor(self) -> Module:
        pass