from typing import List, Callable

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss


class PixelLossOnThreeImagesLoss(PoserGanLoss):
    def __init__(self,
                 weights: List[float],
                 device=torch.device('cpu')):
        self.weights = weights
        self.device = device

    def discriminator_loss(self, G: Module, D: Module, batch: List[Tensor],
                           log_func: Callable[[str, float], None] = None) -> Tensor:
        return torch.zeros(1, device=self.device)

    def generator_loss(self, G: Module, D: Module, batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]
        generated_images = G(source_image, pose)

        image_0_loss = self.weights[0] * (generated_images[0] - real_target_image).abs().mean()
        image_1_loss = self.weights[1] * (generated_images[1] - real_target_image).abs().mean()
        image_2_loss = self.weights[2] * (generated_images[2] - real_target_image).abs().mean()
        loss = image_0_loss + image_1_loss + image_2_loss

        if log_func is not None:
            log_func("generator_image_0_loss", image_0_loss.item())
            log_func("generator_image_1_loss", image_1_loss.item())
            log_func("generator_image_2_loss", image_2_loss.item())
            log_func("generator_loss", loss.item())

        return loss
