import abc

from gans.gan_module import GanModule


class StyleGan:
    __metaclass__ = abc.ABC

    @property
    @abc.abstractmethod
    def latent_vector_size(self) -> int:
        pass

    @abc.abstractmethod
    def mapping_module(self) -> GanModule:
        pass

    @abc.abstractmethod
    def generator_module_stabilize(self, image_size) -> GanModule:
        pass

    @abc.abstractmethod
    def discriminator_stabilize(self, image_size) -> GanModule:
        pass

    @abc.abstractmethod
    def generator_module_transition(self, image_size) -> GanModule:
        pass

    @abc.abstractmethod
    def discriminator_transition(self, image_size) -> GanModule:
        pass