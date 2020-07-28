import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, Tanh, Sigmoid
from torch.nn.functional import affine_grid, grid_sample

from hana.rindou.nn.image_generator_bodies import bottleneck_generator_body, UNetGeneratorBody
from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class Clamp(Module):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: Tensor):
        return x.clamp(self.min_value, self.max_value)


class PumarolaAndApperanceFlowGenerator(PoserGanModule):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 body_type: str = 'bottleneck',
                 align_corners: bool = True):
        super().__init__()
        self.align_corners = align_corners

        if body_type == 'bottleneck':
            self.puaf_feature_xform = bottleneck_generator_body(
                image_size, 4 + pose_size, initial_dim,
                bottleneck_image_size, bottleneck_block_count,
                initialization_method)
        elif body_type == "unet":
            self.puaf_feature_xform = UNetGeneratorBody(
                image_size, 4 + pose_size, initial_dim,
                bottleneck_image_size, bottleneck_block_count,
                initialization_method)
        else:
            raise RuntimeError("Invalid body_type: %s" % body_type)

        init = create_init_function(initialization_method)
        self.puaf_color_change = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Tanh())
        self.puaf_alpha_mask = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Sigmoid())
        self.puaf_grid_change = init(Conv2d(initial_dim, 2, kernel_size=7, stride=1, padding=3, bias=False))

    def forward(self, image: Tensor, pose: Tensor):
        n = image.size(0)
        c = image.size(1)
        h = image.size(2)
        w = image.size(3)

        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))
        x = torch.cat([image, pose], dim=1)
        puaf_xformed = self.puaf_feature_xform(x)

        color_change = self.puaf_color_change(puaf_xformed)
        alpha_mask = self.puaf_alpha_mask(puaf_xformed)
        color_changed = alpha_mask * image + (1 - alpha_mask) * color_change

        grid_change = torch.transpose(self.puaf_grid_change(puaf_xformed).view(n, 2, h * w), 1, 2).view(n, h, w,
                                                                                                        2)
        device = self.puaf_grid_change.weight.device
        identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=self.align_corners)
        grid = base_grid + grid_change
        resampled = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=self.align_corners)

        return [color_changed, resampled, color_change, alpha_mask, grid_change, grid]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])


class PumarolaAndAppearanceFlowGeneratorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 requires_optimization: bool = True,
                 body_type: str = 'bottleneck',
                 align_corners: bool = True):
        self._image_size = image_size
        self._pose_size = pose_size
        self._initial_dim = initial_dim
        self._bottleneck_image_size = bottleneck_image_size
        self._bottleneck_block_count = bottleneck_block_count
        self._initialization_method = initialization_method
        self._requires_optimization = requires_optimization
        self._body_type = body_type
        self._align_corners = align_corners

    def requires_optimization(self) -> bool:
        return self._requires_optimization

    def get_module(self) -> PoserGanModule:
        return PumarolaAndApperanceFlowGenerator(
            image_size=self._image_size,
            pose_size=self._pose_size,
            initial_dim=self._initial_dim,
            bottleneck_image_size=self._bottleneck_image_size,
            bottleneck_block_count=self._bottleneck_block_count,
            initialization_method=self._initialization_method,
            body_type=self._body_type,
            align_corners=self._align_corners)


if __name__ == "__main__":
    module = PumarolaAndApperanceFlowGenerator()
    state_dict = module.state_dict()
    # i = 0
    # for key in state_dict:
    #    print(i, "--", key, state_dict[key].shape)
    #    i += 1
    for key in state_dict:
        print("\"%s\"," % key)
