from typing import List, Dict, Callable

from torch import Tensor
from torch.nn import CrossEntropyLoss

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class MulticlassCrossEntropyLoss(PoserGanLoss):
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.loss = CrossEntropyLoss()

    def compute_with_cached_outputs(self,
                                    G: PoserGanModule,
                                    D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        labels = batch[-1]
        G_output = self.get_G_output(G, batch, outputs)
        loss = self.weight * self.loss(G_output, labels)
        if log_func is not None:
            log_func("cross_entropy", loss.item())
        return loss
