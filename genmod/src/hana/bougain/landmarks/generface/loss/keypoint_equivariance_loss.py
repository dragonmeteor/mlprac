from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.bougain.landmarks.generface.loss.keypoint_based_face_morpher_loss import KeypointBasedFaceMorpherLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class KeypointEquivarianceLoss(KeypointBasedFaceMorpherLoss):
    def __init__(self,
                 which_image_to_morph: str = 'original',
                 weight: float = 1.0):
        super().__init__(which_image_to_morph)
        self.weight = weight

    def compute_with_cached_outputs(self,
                                    G: PoserGanModule,
                                    D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        original_keypoints = self.get_original_keypoint(G, batch, outputs)
        xformed_augmented_keypoints = self.get_xformed_augmented_keypoint(G, batch, outputs)
        loss = (original_keypoints - xformed_augmented_keypoints).abs().mean() * self.weight
        if log_func is not None:
            log_func("loss", loss.item())
        return loss
