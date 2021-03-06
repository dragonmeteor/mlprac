from typing import List, Dict, Callable

import torch
from torch import Tensor

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


def average_score(x: Tensor):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    assert c == 1
    return x.view(n,c,h*w).mean(dim=2).squeeze()

class FirstNOutputWganForGeneratorLoss(PoserGanLoss):
    def __init__(self, weights: List[float]):
        self.weights = weights

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule,
                                    batch: List[Tensor], outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        G_output = self.get_G_output(G, batch, outputs)
        device = source_image.device

        losses = []
        for i in range(len(self.weights)):
            weight = self.weights[i]
            if weight > 0:
                fake_score = average_score(D(source_image, pose, G_output[i])[0])
                image_loss = -fake_score.mean() * weight
            else:
                image_loss = torch.zeros(1, device=device)
            losses.append(image_loss)

        loss = torch.zeros(1, device=device)
        for image_loss in losses:
            loss += image_loss

        if log_func is not None:
            for i in range(len(losses)):
                log_func("loss_%03d" % i, losses[i].item())
            log_func("loss", loss.item())

        return loss