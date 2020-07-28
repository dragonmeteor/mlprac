from typing import List, Dict, Callable

from torch import Tensor

from hana.bougain.landmarks.generface.loss.keypoint_based_face_morpher_loss import KeypointBasedFaceMorpherLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class KeypointBasedFaceMorpher01L1Loss(KeypointBasedFaceMorpherLoss):
    def __init__(self, weight: float = 1.0, which_image_to_morph: str = 'original'):
        super().__init__(which_image_to_morph)
        self.weight = weight

    def compute_with_cached_outputs(self,
                                    G: PoserGanModule,
                                    D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        target_image = self.get_input_target_image(batch)
        G_target_image = self.get_output_morphed_image(G, batch, outputs)
        loss = (target_image - G_target_image).abs().mean() * self.weight
        if log_func is not None:
            log_func("loss", loss.item())
        return loss
