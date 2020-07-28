from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class KeypointBasedFaceMorpher00L1Loss(PoserGanLoss):
    def __init__(self, diff_image_weight: float = 256.0):
        super().__init__()
        self.diff_image_weight = diff_image_weight

    def get_G_output(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        if "G_output" in outputs:
            G_output = outputs["G_output"]
        else:
            G_output = G.forward_from_batch(batch)
            outputs["G_output"] = G_output
        return G_output

    def compute_with_cached_outputs(self,
                                    G: PoserGanModule,
                                    D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        target_image = batch[1]
        diff_image = source_image - target_image
        G_output = self.get_G_output(G, batch, outputs)
        G_source_image = G_output[0]
        G_target_image = G_output[1]
        G_diff_image = G_source_image - G_target_image

        source_image_loss = (source_image - G_source_image).abs().mean()
        target_image_loss = (target_image - G_target_image).abs().mean()
        diff_image_loss = (diff_image - G_diff_image).abs().mean() * self.diff_image_weight
        loss = source_image_loss + target_image_loss + diff_image_loss

        if log_func is not None:
            log_func("source_image_loss", source_image_loss.item())
            log_func("target_image_loss", target_image_loss.item())
            log_func("diff_image_loss", diff_image_loss.item())
            log_func("loss", loss.item())

        return loss
