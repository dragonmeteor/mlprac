from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.bougain.landmarks.generface.loss.keypoint_based_face_morpher_loss import KeypointBasedFaceMorpherLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class HeatmapOverlapLoss(KeypointBasedFaceMorpherLoss):
    def __init__(self, which_image_to_morph: str = 'original', weight: float = 1.0):
        super().__init__(which_image_to_morph)
        self.weight = weight

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule, batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        G_output = self.get_G_output(G, batch, outputs)
        heatmap = G_output[self.G_output_heatmap_index]
        n, k, w, h = heatmap.shape
        # sum2 = (heatmap ** 2).sum(dim=1)
        # sum = heatmap.sum(dim=1)
        # loss = self.weight * (sum ** 2 - sum2).mean() / (k * k)
        heatmap_max = heatmap.max(dim=1, keepdim=True)[0]
        loss = torch.topk(heatmap / heatmap_max, 2, dim=1)[0][:, 1, :, :].mean()
        if log_func is not None:
            log_func("loss", loss.item())
        return loss
