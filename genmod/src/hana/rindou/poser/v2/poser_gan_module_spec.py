import abc

from torch.nn import Module

from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class PoserGanModuleSpec:
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def requires_optimization(self) -> bool:
		pass

	@abc.abstractmethod
	def get_module(self) -> PoserGanModule:
		pass