from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.rindou.poser.v2.loss.first_n_output_ls_gan_for_generator_loss import image_average
from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class TwoOutputsLsGanForDiscriminatorLoss(PoserGanLoss):
    """
    Used to train a GAN where the generator output two images.
    """
    def __init__(self,
                 weight: float = 1.0,
                 batch_source_image_index: int = 0,
                 batch_pose_index: int = 1,
                 batch_real_image_index: int = 2):
        self.weight = weight
        self.batch_source_image_index = batch_source_image_index
        self.batch_pose_index = batch_pose_index
        self.batch_real_image_index = batch_real_image_index

    def get_source_image_from_batch(self, batch):
        return batch[self.batch_source_image_index]

    def get_pose_from_batch(self, batch):
        return batch[self.batch_pose_index]

    def get_real_image_from_batch(self, batch):
        return batch[self.batch_real_image_index]

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = self.get_source_image_from_batch(batch)
        pose = self.get_pose_from_batch(batch)
        real_target_image = self.get_real_image_from_batch(batch)
        G_output = self.get_G_output(G, batch, outputs)
        device = batch[0].device
        real_score = D(source_image, pose, real_target_image, real_target_image)[0]
        real_score_loss = image_average((real_score - 1) ** 2).mean()

        if self.weight > 0:
            fake_score = D(source_image, pose, G_output[0], G_output[1])[0]
            loss = (real_score_loss + image_average(fake_score ** 2).mean()) * self.weight
        else:
            loss = torch.zeros(1, device=device)

        if log_func is not None:
            log_func("loss", loss.item())

        return loss
