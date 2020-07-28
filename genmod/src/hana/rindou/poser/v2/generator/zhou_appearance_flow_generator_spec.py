from typing import List

import torch
from torch import Tensor
from torch.nn.functional import affine_grid, grid_sample

from hana.rindou.nn2.zhou_module import ZhouModule
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class ZhouAppearanceFlowGenerator(PoserGanModule):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 initial_channels: int = 64,
                 image_repr_size: int = 4096,
                 pose_repr_sizes: List[int] = None,
                 max_channel: int = 512,
                 initialization_method: str = 'he',
                 align_corners: bool = False):
        super().__init__()
        self.body = ZhouModule(
            image_size,
            image_channels,
            pose_size,
            initial_channels,
            image_repr_size,
            pose_repr_sizes,
            max_channel,
            initialization_method)
        self.align_corners = align_corners

    def forward(self, image: Tensor, pose: Tensor):
        flow = self.body(image, pose)
        n, c, h, w = flow.shape
        assert c == 2
        grid_change = torch.transpose(flow.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        device = grid_change.device
        identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=self.align_corners)
        grid = base_grid + grid_change
        resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border',
                                      align_corners=self.align_corners)

        return [resampled_image, grid_change]

    def forward_from_batch(self, batch):
        image = batch[0]
        pose = batch[1]
        return self.forward(image, pose)


class ZhouAppearanceFlowGeneratorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 initial_channels: int = 64,
                 image_repr_size: int = 4096,
                 pose_repr_sizes: List[int] = None,
                 max_channel: int = 512,
                 initialization_method: str = 'he',
                 align_corners: bool = False):
        self.align_corners = align_corners
        self.initialization_method = initialization_method
        self.max_channel = max_channel
        self.pose_repr_sizes = pose_repr_sizes
        self.image_repr_size = image_repr_size
        self.initial_channels = initial_channels
        self.pose_size = pose_size
        self.image_channels = image_channels
        self.image_size = image_size

    def requires_optimization(self) -> bool:
        return True

    def get_module(self) -> PoserGanModule:
        return ZhouAppearanceFlowGenerator(
            self.image_size,
            self.image_channels,
            self.pose_size,
            self.initial_channels,
            self.image_repr_size,
            self.pose_repr_sizes,
            self.max_channel,
            self.initialization_method,
            self.align_corners)
