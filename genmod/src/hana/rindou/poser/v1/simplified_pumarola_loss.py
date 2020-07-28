from typing import List, Callable

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss


class SimplifiedPumarolaLoss(PoserGanLoss):
    def __init__(self,
                 grad_loss_weight=10.0,
                 drift_weight=1e-3,
                 pixel_loss_weight: float = 100.0,
                 pose_loss_weight=4000.0,
                 device=torch.device('cuda')):
        self.grad_loss_weight = grad_loss_weight
        self.drift_weight = drift_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.pose_loss_weight = pose_loss_weight
        self.device = device

    def discriminator_loss(self,
                           G: Module, D: Module,
                           batch: List[Tensor],
                           log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]

        real_score, real_pose = D(source_image, real_target_image)
        fake_target_image = G(source_image, pose)[0]
        fake_score, fake_pose = D(source_image, fake_target_image)

        gan_loss = -real_score.mean() + fake_score.mean()
        wgan_loss = self.wgan_loss(D, source_image, real_target_image, fake_target_image) * self.grad_loss_weight
        drift_loss = self.drift_weight * real_score.mul(real_score).mean()
        pose_loss = self.pose_loss_weight * ((real_pose - pose).abs().mean())
        return gan_loss + wgan_loss + drift_loss + pose_loss

    def generator_loss(self,
                       G: Module, D: Module,
                       batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        real_target_image = batch[2]

        fake_target_image = G(source_image, pose)[0]
        fake_score, fake_pose = D(source_image, fake_target_image)

        return self.generator_loss_calculation_only(
            source_image,
            pose,
            real_target_image,
            fake_target_image,
            fake_score,
            fake_pose)

    def generator_loss_calculation_only(self,
                                        source_image: Tensor,
                                        pose: Tensor,
                                        real_target_image: Tensor,
                                        fake_target_image: Tensor,
                                        fake_score: Tensor,
                                        fake_pose):
        pose_diff = fake_pose - pose
        pose_loss = self.pose_loss_weight * pose_diff.mul(pose_diff).mean()

        if self.pixel_loss_weight > 0:
            pixel_loss = self.pixel_loss_weight * \
                         (fake_target_image - real_target_image).abs().mean()
        else:
            pixel_loss = 0.0

        return -fake_score.mean() + pose_loss + pixel_loss

    def wgan_loss(self, D: Module, source_image, real_target_image, fake_target_image):
        n = real_target_image.shape[0]
        sample_size = 1
        for i in range(1, len(real_target_image.shape)):
            sample_size *= real_target_image.shape[i]

        interpolates = self.create_interpolates(real_target_image, fake_target_image)
        interpolates.requires_grad_(True)
        grad_outputs = torch.ones(n, device=self.device)
        D_interp_score, D_interp_pose = D(source_image, interpolates)
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
