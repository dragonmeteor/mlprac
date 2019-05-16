import abc

from gans.gan_module import GanModule


class PgGan:
    __metaclass__ = abc.ABC

    @property
    @abc.abstractmethod
    def latent_vector_size(self) -> int:
        pass

    @abc.abstractmethod
    def generator_stabilize(self, image_size: int) -> GanModule:
        pass

    @abc.abstractmethod
    def generator_transition(self, image_size: int) -> GanModule:
        pass

    @abc.abstractmethod
    def discriminator_stabilize(self, image_size: int) -> GanModule:
        pass

    @abc.abstractmethod
    def discriminator_transition(self, image_size: int) -> GanModule:
        pass