from typing import List, Callable, Dict

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss


class FirstNOutputTotalVariationLoss(PoserGanLoss):
    def __init__(self, weights: List[float]):
        self.weights = weights

    def compute_with_cached_outputs(self, G: Module, D: Module, batch: List[Tensor], outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        G_output = self.get_G_output(G, batch, outputs)
        device = G_output[0].device

        losses = []
        loss = torch.zeros(1, device=device)

        for i in range(len(self.weights)):
            image = G_output[i]
            n = image.shape[0]
            c = image.shape[1]
            h = image.shape[2]
            w = image.shape[3]
            didx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().sum()
            didy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().sum()
            image_loss = self.weights[i] * 1.0 / (n*c*h*w) * (didx * didy)
            losses.append(image_loss)
            loss += image_loss

        if log_func is not None:
            for i in range(len(self.weights)):
                log_func("loss_%03d" % i, losses[i].item())
            log_func("loss", loss.item())

        return loss
