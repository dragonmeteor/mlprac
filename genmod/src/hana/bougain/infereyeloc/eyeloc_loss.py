from modulefinder import Module
from typing import List, Dict, Callable

from torch import Tensor

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss


class EyeLocL2Loss(PoserGanLoss):
    def __init__(self, weight:float=100.0):
        self.weight = weight

    def compute_with_cached_outputs(self,
                                    G: Module,
                                    D: Module,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        real_eyelocs = batch[-1]
        G_output = self.get_G_output(G, batch, outputs)

        inferred_eyelocs = G_output[0]
        loss = ((real_eyelocs - inferred_eyelocs) ** 2).sum(dim=1).mean() * self.weight
        if log_func is not None:
            log_func("loss", loss.item())

        return loss


class EyeLocL1Loss(PoserGanLoss):
    def __init__(self, weight:float=100.0):
        self.weight = weight

    def compute_with_cached_outputs(self,
                                    G: Module,
                                    D: Module,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        real_eyelocs = batch[-1]
        G_output = self.get_G_output(G, batch, outputs)

        inferred_eyelocs = G_output[0]
        loss = (real_eyelocs - inferred_eyelocs).abs().sum(dim=1).mean() * self.weight
        if log_func is not None:
            log_func("loss", loss.item())

        return loss
