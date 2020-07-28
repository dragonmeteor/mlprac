from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


def image_average(x: Tensor):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    assert c == 1
    return x.view(n, c, h * w).mean(dim=2).squeeze()


class TwoCopiesLsGanForGeneratorLoss(PoserGanLoss):
    """
    Used to train a GAN where the generator output two images, and the discriminator also output two reality scores.
    """

    def __init__(self,
                 weight: float = 1.0,
                 batch_source_image_index: int = 0,
                 batch_pose_index: int = 1):
        self.weight = weight
        self.batch_source_image_index = batch_source_image_index
        self.batch_pose_index = batch_pose_index

    def get_source_image_from_batch(self, batch):
        return batch[self.batch_source_image_index]

    def get_pose_from_batch(self, batch):
        return batch[self.batch_pose_index]

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule,
                                    batch: List[Tensor], outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = self.get_source_image_from_batch(batch)
        pose = self.get_pose_from_batch(batch)
        G_output = self.get_G_output(G, batch, outputs)
        device = source_image.device

        if self.weight > 0:
            fake_scores = D(source_image, pose, G_output[0], G_output[1])
            losses = [image_average((fake_score - 1.0) ** 2).mean() * self.weight for fake_score in fake_scores]
        else:
            losses = [torch.zeros(1, device=device), torch.zeros(1, device=device)]

        loss = losses[0] + losses[1]
        if log_func is not None:
            log_func("loss", loss.item())
            log_func("loss_000", losses[0].item())
            log_func("loss_001", losses[1].item())

        return loss
