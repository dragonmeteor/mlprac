from abc import ABC
from typing import List, Callable, Dict

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class ConstantLoss(PoserGanLoss):
    def compute(self, G: PoserGanModule, D: PoserGanModule, batch: List[Tensor],
                log_func: Callable[[str, float], None] = None) -> Tensor:
        return torch.zeros(1, device=batch[0].device)

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule,
                                    batch: List[Tensor], outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        return torch.zeros(1, device=batch[0].device)
