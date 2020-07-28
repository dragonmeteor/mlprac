from typing import List, Callable

import torch
from torch.nn import Module
from torch import Tensor

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.regressor.pose_regressor_spec import PoseRegressorSpec
from hana.rindou.util import torch_load


class PumarolaPixelAndFixedPoseRegressorLoss(PoserGanLoss):
    def __init__(self,
                 regressor_file_name: str,
                 regressor_spec: PoseRegressorSpec,
                 pixel_loss_weight: float = 100.0,
                 pose_loss_weight=4000.0,
                 alpha_magnitude_weight=0.1,
                 alpha_smoothness_weight=0.0001,
                 device=torch.device('cuda')):
        super().__init__()
        self.pixel_loss_weight = pixel_loss_weight
        self.pose_loss_weight = pose_loss_weight
        self.alpha_magnitude_weight = alpha_magnitude_weight
        self.alpha_smoothness_weight = alpha_smoothness_weight
        self.device = device
        self.regressor_file_name = regressor_file_name
        self.regressor_spec = regressor_spec

        self.regressor = None

    def get_regressor(self):
        if self.regressor is None:
            R = self.regressor_spec.regressor().to(self.device)
            self.regressor = R
            R.load_state_dict(torch_load(self.regressor_file_name))
        self.regressor.train(False)
        return self.regressor

    def discriminator_loss(self, G: Module, D: Module, batch: List[Tensor],
                           log_func: Callable[[str, float], None] = None) -> Tensor:
        return torch.zeros(1, device=self.device)

    def generator_loss(self, G: Module, D: Module, batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]

        fake_target_image = G(source_image, pose)[0]
        pixel_loss = self.pixel_loss_weight * (fake_target_image - real_target_image).abs().mean()

        if self.pose_loss_weight > 0:
            R = self.get_regressor()
            fake_pose = R(source_image, fake_target_image)
            pose_diff = fake_pose - pose
            pose_loss = self.pose_loss_weight * pose_diff.abs().mean()
        else:
            pose_loss = 0

        loss = pose_loss + pixel_loss

        if log_func is not None:
            if self.pose_loss_weight > 0:
                log_func("generator_pose_loss", pose_loss.item())
            log_func("generator_pixel_loss", pixel_loss.item())
            log_func("generator_loss", loss.item())

        return loss
