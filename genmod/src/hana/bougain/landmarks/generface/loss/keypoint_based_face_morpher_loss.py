from abc import ABC
from typing import List, Dict

import torch
from torch import Tensor

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule

ORIGINAL = 'original'
G_OUTPUT = "G_output"
KEYPOINT_OUTPUT = "keypoint_output"


class KeypointBasedFaceMorpherLoss(PoserGanLoss, ABC):
    def __init__(self,
                 which_image_to_morph: str = ORIGINAL,
                 original_source_image_index: int = 0,
                 original_target_image_index: int = 1,
                 augmented_source_image_index: int = 2,
                 augmented_target_image_index: int = 3,
                 matrix_index: int = 4,
                 inverse_matrix_index: int = 5,
                 G_output_morphed_image_index: int = 0,
                 G_output_keypoint_index: int = 3,
                 G_output_heatmap_index: int = 4,
                 keypoint_output_keypoint_index: int = 0):
        super().__init__()
        self.G_output_morphed_image_index = G_output_morphed_image_index
        self.keypoint_output_keypoint_index = keypoint_output_keypoint_index
        self.G_output_heatmap_index = G_output_heatmap_index
        self.G_output_keypoint_index = G_output_keypoint_index
        self.which_image_to_morph = which_image_to_morph
        self.inverse_matrix_index = inverse_matrix_index
        self.matrix_index = matrix_index
        self.augmented_target_image_index = augmented_target_image_index
        self.augmented_source_image_index = augmented_source_image_index
        self.original_target_image_index = original_target_image_index
        self.original_source_image_index = original_source_image_index

    def get_G_output(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        if G_OUTPUT in outputs:
            G_output = outputs[G_OUTPUT]
        else:
            G_output = G.forward_from_batch([self.get_input_source_image(batch), self.get_input_target_image(batch)])
            outputs[G_OUTPUT] = G_output
        return G_output

    def get_keypoint_output(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        if KEYPOINT_OUTPUT in outputs:
            keypoint_output = outputs[KEYPOINT_OUTPUT]
        else:
            if self.which_image_to_morph == "original":
                target_image = batch[self.augmented_target_image_index]
            else:
                target_image = batch[self.original_target_image_index]
            keypoint_output = G.keypoint_detector.forward(target_image)
            outputs[KEYPOINT_OUTPUT] = keypoint_output
        return keypoint_output

    def get_original_keypoint(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        if self.which_image_to_morph == ORIGINAL:
            return self.get_G_output(G, batch, outputs)[self.G_output_keypoint_index]
        else:
            return self.get_keypoint_output(G, batch, outputs)[self.keypoint_output_keypoint_index]

    def get_augmented_keypoint(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        if self.which_image_to_morph == ORIGINAL:
            return self.get_keypoint_output(G, batch, outputs)[self.keypoint_output_keypoint_index]
        else:
            return self.get_G_output(G, batch, outputs)[self.G_output_keypoint_index]

    def get_xformed_augmented_keypoint(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        augmented_keypoints = self.get_augmented_keypoint(G, batch, outputs)
        device = augmented_keypoints.device
        n, k, _ = augmented_keypoints.shape
        matrices = batch[self.matrix_index].unsqueeze(1)
        augmented_keypoints = torch.cat([augmented_keypoints, torch.ones(n, k, 1, device=device)], dim=2) \
            .view(n, k, 3, 1)
        xformed_augmented_keypoints = torch.matmul(matrices, augmented_keypoints).view(n, k, 2)
        return xformed_augmented_keypoints

    def get_matrix(self, batch: List[Tensor]):
        return batch[self.matrix_index]

    def get_input_source_image(self, batch: List[Tensor]):
        if self.which_image_to_morph == ORIGINAL:
            return batch[self.original_source_image_index]
        else:
            return batch[self.augmented_source_image_index]

    def get_input_target_image(self, batch: List[Tensor]):
        if self.which_image_to_morph == ORIGINAL:
            return batch[self.original_target_image_index]
        else:
            return batch[self.augmented_target_image_index]

    def get_output_morphed_image(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        return self.get_G_output(G, batch, outputs)[self.G_output_morphed_image_index]
