from typing import List, Dict, Callable

import torch
from torch import Tensor
from torch.nn.functional import affine_grid

from hana.bougain.landmarks.generface.loss.keypoint_based_face_morpher_loss import KeypointBasedFaceMorpherLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class KeypointConcentrationLoss(KeypointBasedFaceMorpherLoss):
    def __init__(self,
                 which_image_to_morph: str = 'original',
                 weight=1.0):
        super().__init__(which_image_to_morph)
        self.weight = weight

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule, batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        G_output = self.get_G_output(G, batch, outputs)
        keypoint = G_output[self.G_output_keypoint_index]
        heatmap = G_output[self.G_output_heatmap_index]
        n, k, h, w = heatmap.shape
        heatmap = heatmap.view(n, k, h * w, 1)

        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).to(heatmap.device).unsqueeze(0).repeat(n, 1, 1)
        grid = affine_grid(identity, [n, k, h, w], align_corners=False).view(n, 1, h * w, 2).repeat(1, k, 1, 1)
        loss = ((grid - keypoint.view(n, k, 1, 2)) ** 2 * heatmap).sum(dim=(2, 3)).mean() * self.weight

        if log_func is not None:
            log_func("loss", loss.item())

        return loss
