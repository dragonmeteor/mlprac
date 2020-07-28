from torch.nn import Module

from hana.rindou.poser.v1.pumarola import PumarolaGenerator
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class PumarolaGeneratorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 requires_optimization: bool = True,
                 body_type: str = 'bottleneck'):
        self._image_size = image_size
        self._pose_size = pose_size
        self._initial_dim = initial_dim
        self._bottleneck_image_size = bottleneck_image_size
        self._bottleneck_block_count = bottleneck_block_count
        self._initialization_method = initialization_method
        self._requires_optimization = requires_optimization
        self._body_type = body_type

    def requires_optimization(self) -> bool:
        return self._requires_optimization

    def get_module(self) -> PoserGanModule:
        return PumarolaGenerator(
            image_size=self._image_size,
            pose_size=self._pose_size,
            initial_dim=self._initial_dim,
            bottleneck_image_size=self._bottleneck_image_size,
            bottleneck_block_count=self._bottleneck_block_count,
            initialization_method=self._initialization_method,
            body_type=self._body_type)
