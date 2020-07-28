from typing import List, Callable

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v1.simplified_pumarola_loss import SimplifiedPumarolaLoss


class PumarolaLoss(PoserGanLoss):
    def __init__(self,
                 grad_loss_weight=10.0,
                 drift_weight=1e-3,
                 pixel_loss_weight: float = 100.0,
                 pose_loss_weight=4000.0,
                 alpha_magnitude_weight=0.1,
                 alpha_smoothness_weight=0.0001,
                 device=torch.device('cuda')):
        super().__init__()
        self.grad_loss_weight = grad_loss_weight
        self.drift_weight = drift_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.pose_loss_weight = pose_loss_weight
        self.alpha_magnitude_weight = alpha_magnitude_weight
        self.alpha_smoothness_weight = alpha_smoothness_weight
        self.device = device
        self.simplified_pumarola = SimplifiedPumarolaLoss(grad_loss_weight,
                                                          drift_weight,
                                                          pixel_loss_weight,
                                                          pose_loss_weight,
                                                          device)

    def generator_loss(self,
                       G: Module, D: Module,
                       batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]

        fake_target_image, fake_alpha = G(source_image, pose)
        fake_score, fake_pose = D(source_image, fake_target_image)

        simplified_loss = self.simplified_pumarola.generator_loss_calculation_only(
            source_image,
            pose,
            real_target_image,
            fake_target_image,
            fake_score,
            fake_pose)

        sample_size = 1
        for i in range(1, len(source_image.shape)):
            sample_size *= source_image.shape[i]

        alpha_magnitude_loss = fake_alpha.mean() * self.alpha_magnitude_weight
        alpha_smoothness_loss = self.alpha_smoothness_weight * \
                                ((fake_alpha[:, :, 1:, :] - fake_alpha[:, :, :-1, :]).abs().mean() + \
                                 (fake_alpha[:, :, :, 1:] - fake_alpha[:, :, :, :-1]).abs().mean())

        return simplified_loss + alpha_magnitude_loss + alpha_smoothness_loss

    def discriminator_loss(self,
                           G: Module, D: Module,
                           batch: List[Tensor],
                           log_func: Callable[[str, float], None] = None) -> Tensor:
        return self.simplified_pumarola.discriminator_loss(G, D, batch, log_func)
