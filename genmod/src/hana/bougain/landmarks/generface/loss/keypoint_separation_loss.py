from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.bougain.landmarks.generface.loss.keypoint_based_face_morpher_loss import KeypointBasedFaceMorpherLoss
from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


def keypoint_separtion_loss(keypoints, sigma):
    n, k, _ = keypoints.shape
    keypoints = keypoints.permute((0, 2, 1))  # [n,2,k]

    keypoints_dim_2 = keypoints.unsqueeze(3).permute((0, 1, 3, 2)).repeat(1, 1, k, 1)
    keypoints_dim_3 = keypoints.unsqueeze(3).repeat(1, 1, 1, k)
    keypoints_diff = keypoints_dim_2 - keypoints_dim_3
    keypoints_distance = (keypoints_diff ** 2).sum(dim=1)
    keypoints_gaussian = torch.exp(-keypoints_distance / (2 * sigma * sigma))
    loss = (keypoints_gaussian.sum(dim=(1, 2)) - k).mean()

    return loss


class KeypointSeparationLoss(KeypointBasedFaceMorpherLoss):
    def __init__(self,
                 which_image_to_morph: str = 'original',
                 weight=1.0,
                 sigma=3.0 / 256.0):
        super().__init__(which_image_to_morph)
        self.sigma = sigma
        self.weight = weight

    def compute_with_cached_outputs(self,
                                    G: PoserGanModule,
                                    D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        if self.which_image_to_morph == 'original':
            keypoints = self.get_original_keypoint(G, batch, outputs)
        else:
            keypoints = self.get_xformed_augmented_keypoint(G, batch, outputs)
        loss = keypoint_separtion_loss(keypoints, self.sigma) * self.weight
        if log_func is not None:
            log_func("loss", loss.item())
        return loss


if __name__ == "__main__":
    keypoints = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    loss = keypoint_separtion_loss(keypoints.unsqueeze(0), 1.0)
    print(loss)
