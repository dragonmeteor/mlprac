from typing import List, Callable, Dict

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss


class FirstNOutputsDiffFromTargetL1Loss(PoserGanLoss):
    def __init__(self, weights: List[float]):
        self.weights = weights

    def compute_with_cached_outputs(self, G: Module, D: Module, batch: List[Tensor], outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        target_image = batch[-1]
        G_output = self.get_G_output(G, batch, outputs)
        device = target_image.device

        losses = []
        loss = torch.zeros(1, device=device)
        for i in range(len(self.weights)):
            image_loss = self.weights[i] * (G_output[i] - target_image).abs().mean()
            losses.append(image_loss)
            loss += image_loss

        if log_func is not None:
            for i in range(len(self.weights)):
                log_func("loss_%03d" % i, losses[i].item())
            log_func("loss", loss.item())

        return loss
