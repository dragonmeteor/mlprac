from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.rindou.poser.v2.loss.first_n_output_wgan_for_generator_loss import average_score
from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class FirstNOutputWganGpForDiscriminatorLoss(PoserGanLoss):
    def __init__(self,
                 weights: List[float],
                 gradient_weight: float,
                 drift_weight: float):
        self.wgan_loss_weights = weights
        self.gradient_weight = gradient_weight
        self.drift_weight = drift_weight

    def get_source_image_from_batch(self, batch):
        return batch[0]

    def get_pose_from_batch(self, batch):
        return batch[1]

    def get_real_image_from_batch(self, batch):
        return batch[2]

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule,
                                    batch: List[Tensor], outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = self.get_source_image_from_batch(batch)
        pose = self.get_pose_from_batch(batch)
        real_target_image = self.get_real_image_from_batch(batch)
        G_output = self.get_G_output(G, batch, outputs)
        device = source_image.device
        real_score = average_score(D(source_image, pose, real_target_image)[0])

        losses = []
        for i in range(len(self.wgan_loss_weights)):
            weight = self.wgan_loss_weights[i]
            if weight > 0:
                fake_target_image = G_output[i]
                fake_score = average_score(D(source_image, pose, fake_target_image)[0])
                wgan_loss = real_score.mean() - fake_score.mean()
                gp_loss = self.gradient_loss(D, source_image, pose, real_target_image, fake_target_image)
                drift_loss = self.drift_weight * real_score.mul(real_score).mean()
                image_loss = weight * (wgan_loss + gp_loss + drift_loss)
            else:
                image_loss = torch.zeros(1, device=device)
            losses.append(image_loss)

        loss = torch.zeros(1, device=device)
        for image_loss in losses:
            loss += image_loss

        if log_func is not None:
            for i in range(len(losses)):
                log_func("loss_%03d" % i, losses[i].item())
            log_func("loss", loss.item())

        return loss

    def gradient_loss(self, D: PoserGanModule, source_image, pose, real_target_image, fake_target_image):
        device = source_image.device
        n = real_target_image.shape[0]
        sample_size = 1
        for i in range(1, len(real_target_image.shape)):
            sample_size *= real_target_image.shape[i]

        interpolates = self.create_interpolates(real_target_image, fake_target_image)
        interpolates.requires_grad_(True)
        grad_outputs = torch.ones(n, device=device)
        D_interp_score = average_score(D(source_image, pose, interpolates)[0])
        interpolates_grad = torch.autograd.grad(D_interp_score,
                                                interpolates,
                                                grad_outputs=grad_outputs,
                                                only_inputs=True,
                                                create_graph=True,
                                                retain_graph=True)[0]
        grad_norm = interpolates_grad.view(n, sample_size).norm(2, dim=1)
        grad_diff = grad_norm - 1.0
        return grad_diff.mul(grad_diff).mean() * self.gradient_weight

    def create_interpolates(self, real_image: Tensor, fake_image: Tensor):
        device = real_image.device
        n = real_image.shape[0]
        assert fake_image.shape[0] == n
        alpha_shape = [1 for i in range(len(real_image.shape))]
        alpha_shape[0] = n
        alpha = torch.rand(alpha_shape, device=device)
        return (real_image * alpha + fake_image * (1.0 - alpha)).detach()
