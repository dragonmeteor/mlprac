from typing import List, Tuple, Callable, Dict

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss


class SumLoss(PoserGanLoss):
    def __init__(self, losses: List[Tuple[str, PoserGanLoss]]):
        self.losses = losses

    def compute_with_cached_outputs(self, G: Module, D: Module, batch: List[Tensor], outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        device = batch[0].device
        loss_value = torch.zeros(1, device=device)
        for loss_spec in self.losses:
            loss_name = loss_spec[0]
            loss = loss_spec[1]
            if log_func is not None:
                def loss_log_func(name, value):
                    log_func(loss_name + "_" + name, value)
            else:
                loss_log_func = None
            loss_value += loss.compute_with_cached_outputs(G, D, batch, outputs, loss_log_func)

        if log_func is not None:
            log_func("loss", loss_value.item())

        return loss_value
