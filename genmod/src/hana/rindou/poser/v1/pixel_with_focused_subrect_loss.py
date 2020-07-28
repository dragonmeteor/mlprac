from typing import List, Callable

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss


class PixelWithFocusedSubrectLoss(PoserGanLoss):
    def __init__(self,
                 whole_image_weight,
                 focused_subrect_weight,
                 focused_subrect_x_range=(64,192),
                 focused_subrect_y_range=(64,192),
                 device:torch.device=torch.device('cpu')):
        self.whole_image_weight = whole_image_weight
        self.focused_subrect_weight = focused_subrect_weight
        self.focused_subrect_x_range = focused_subrect_x_range
        self.focused_subrect_y_range = focused_subrect_y_range
        self.device = device

    def discriminator_loss(self, G: Module, D: Module, batch: List[Tensor],
                           log_func: Callable[[str, float], None] = None) -> Tensor:
        return torch.zeros(1, device=self.device)

    def generator_loss(self, G: Module, D: Module, batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]
        fake_target_image = G(source_image, pose)[0]

        whole_image_loss = self.whole_image_weight * (fake_target_image - real_target_image).abs().mean()

        real_subrect = real_target_image[:,:,
                       self.focused_subrect_y_range[0]:self.focused_subrect_y_range[1],
                       self.focused_subrect_x_range[0]:self.focused_subrect_x_range[1]]
        fake_subrect = fake_target_image[:, :,
                       self.focused_subrect_y_range[0]:self.focused_subrect_y_range[1],
                       self.focused_subrect_x_range[0]:self.focused_subrect_x_range[1]]
        subrect_loss = self.focused_subrect_weight * (fake_subrect - real_subrect).abs().mean()

        loss = whole_image_loss + subrect_loss

        if log_func is not None:
            log_func("generator_whole_image_loss", whole_image_loss.item())
            log_func("generator_subrect_loss", subrect_loss.item())
            log_func("generator_loss", loss.item())

        return loss