from torch.nn import Module

from hana.rindou.poser.v2.generator.pumarola_and_appearance_flow_generator_spec import PumarolaAndApperanceFlowGenerator
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class PuafSelectOneGeneratorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 requires_optimization: bool = True,
                 body_type: str = 'bottleneck',
                 selected_index: int = 0,
                 align_corners: bool = True):
        self._image_size = image_size
        self._pose_size = pose_size
        self._initial_dim = initial_dim
        self._bottleneck_image_size = bottleneck_image_size
        self._bottleneck_block_count = bottleneck_block_count
        self._initialization_method = initialization_method
        self._requires_optimization = requires_optimization
        self._body_type = body_type
        self._selected_index = selected_index
        self._align_corners = align_corners

    def requires_optimization(self) -> bool:
        return self._requires_optimization

    def get_module(self) -> PoserGanModule:
        return PuafSelectOneGenerator(
            self._image_size,
            self._pose_size,
            self._initial_dim,
            self._bottleneck_image_size,
            self._bottleneck_block_count,
            self._initialization_method,
            self._body_type,
            self._selected_index,
            self._align_corners)


class PuafSelectOneGenerator(PumarolaAndApperanceFlowGenerator):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 body_type: str = 'bottlenect',
                 selected_index: int = 0,
                 align_corners: bool = True):
        super().__init__(
            image_size, pose_size, initial_dim, bottleneck_image_size, bottleneck_block_count,
            initialization_method, body_type, align_corners)
        self.selected_index = selected_index

    def forward(self, image, pose):
        return [super().forward(image, pose)[self.selected_index]]
