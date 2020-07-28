from torch.nn import Module

from hana.rindou.poser.v1.pumarola import DiscriminatorDoNothing
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class DoNothingDiscriminatorSpec(PoserGanModuleSpec):
    def requires_optimization(self) -> bool:
        return False

    def get_module(self) -> PoserGanModule:
        return DiscriminatorDoNothing()