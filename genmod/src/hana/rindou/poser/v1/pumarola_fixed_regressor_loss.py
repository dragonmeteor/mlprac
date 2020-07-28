from typing import List, Callable

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.regressor.pose_regressor_spec import PoseRegressorSpec
from hana.rindou.util import torch_load


class PumarolaFixedRegressorLoss(PoserGanLoss):
    def __init__(self,
                 regressor_file_name: str,
                 regressor_spec: PoseRegressorSpec,
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
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]

        real_score = D(source_image, real_target_image)
        fake_target_image = G(source_image, pose)[0]
        fake_score = D(source_image, fake_target_image)

        wgan_loss = -real_score.mean() + fake_score.mean()
        gp_loss = self.gp_loss(D, source_image, real_target_image, fake_target_image) * self.grad_loss_weight
        drift_loss = self.drift_weight * real_score.mul(real_score).mean()

        loss = wgan_loss + gp_loss + drift_loss

        if log_func is not None:
            log_func("discriminator_wgan_loss", wgan_loss.item())
            log_func("discriminator_gp_loss", gp_loss.item())
            log_func("discriminator_drift_loss", drift_loss.item())
            log_func("discriminator_adversarial_loss", (wgan_loss + gp_loss + drift_loss).item())
            log_func("discriminator_loss", loss.item())

        return loss

    def gp_loss(self, D: Module, source_image, real_target_image, fake_target_image):
        n = real_target_image.shape[0]
        sample_size = 1
        for i in range(1, len(real_target_image.shape)):
            sample_size *= real_target_image.shape[i]

        interpolates = self.create_interpolates(real_target_image, fake_target_image)
        interpolates.requires_grad_(True)
        grad_outputs = torch.ones(n, device=self.device)
        D_interp_score = D(source_image, interpolates)
        interpolates_grad = torch.autograd.grad(D_interp_score,
                                                interpolates,
                                                grad_outputs=grad_outputs,
                                                only_inputs=True,
                                                create_graph=True,
                                                retain_graph=True)[0]
        grad_norm = interpolates_grad.view(n, sample_size).norm(2, dim=1)
        grad_diff = grad_norm - 1.0
        return grad_diff.mul(grad_diff).mean() * self.grad_loss_weight

    def create_interpolates(self, real_image: Tensor, fake_image: Tensor):
        n = real_image.shape[0]
        assert fake_image.shape[0] == n
        alpha_shape = [1 for i in range(len(real_image.shape))]
        alpha_shape[0] = n
        alpha = torch.rand(alpha_shape, device=self.device)
        return (real_image * alpha + fake_image * (1 - alpha)).detach()

    def generator_loss(self, G: Module, D: Module, batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        R = self.get_regressor()
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]

        fake_target_image = G(source_image, pose)[0]
        fake_score = D(source_image, fake_target_image)
        fake_pose = R(source_image, fake_target_image)
        adversarial_loss = -fake_score.mean()

        pose_diff = fake_pose - pose
        pose_loss = self.pose_loss_weight * pose_diff.abs().mean()
        pixel_loss = self.pixel_loss_weight * (fake_target_image - real_target_image).abs().mean()

        loss = adversarial_loss + pose_loss + pixel_loss

        if log_func is not None:
            log_func("generator_adversarial_loss", adversarial_loss.item())
            log_func("generator_pose_loss", pose_loss.item())
            log_func("generator_pixel_loss", pixel_loss.item())
            log_func("generator_loss", loss.item())

        return loss
