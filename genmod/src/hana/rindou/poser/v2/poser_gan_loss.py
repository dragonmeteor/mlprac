import abc
from typing import List, Callable, Dict

from torch import Tensor

from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class PoserGanLoss:
    __metaclass__ = abc.ABCMeta

    def compute(self,
                G: PoserGanModule,
                D: PoserGanModule,
                batch: List[Tensor],
                log_func: Callable[[str, float], None] = None) -> Tensor:
        return self.compute_with_cached_outputs(G, D, batch, {}, log_func)

    @abc.abstractmethod
    def compute_with_cached_outputs(self,
                                    G: PoserGanModule,
                                    D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        pass

    def get_G_output(self, G: PoserGanModule, batch: List[Tensor], outputs: Dict[str, List[Tensor]]):
        if "G_output" in outputs:
            G_output = outputs["G_output"]
        else:
            G_output = G.forward_from_batch(batch[:-1])
            outputs["G_output"] = G_output
        return G_output