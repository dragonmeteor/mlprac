from typing import List, Callable, Dict

import torch
from torch import Tensor
from torch.nn import Module

from hana.rindou.nn.vgg16 import get_vgg16_perceptual_loss_modules
from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


def compute_gram_matrix(x: Tensor):
    n, c, h, w = x.shape
    xx = x.view(n, c, h * w)
    xxT = x.transpose(1, 2)
    return torch.bmm(xx, xxT) / (h * w)


class FirstNOutputDiffFromTargetPerceptualLoss(PoserGanLoss):
    def __init__(self,
                 vgg16_layers: List[str],
                 content_weights: List[float],
                 style_weights: List[float],
                 device=torch.device('cpu')):
        self.vgg16_layers = vgg16_layers
        self.content_weights = content_weights
        self.style_weights = style_weights
        self.device = device
        self.vgg16_modules = None

    def get_vgg16_modules(self):
        if self.vgg16_modules is None:
            self.vgg16_modules = [module.to(self.device)
                                  for module in get_vgg16_perceptual_loss_modules(self.vgg16_layers)]
            for module in self.vgg16_modules:
                module.train(False)
        return self.vgg16_modules

    def compute_with_cached_outputs(self, G: PoserGanModule, D: PoserGanModule,
                                    batch: List[Tensor],
                                    outputs: Dict[str, List[Tensor]],
                                    log_func: Callable[[str, float], None] = None) -> Tensor:
        target_image = batch[-1]
        G_output = self.get_G_output(G, batch, outputs)

        content_losses = []
        style_losses = []
        for i in range(len(self.content_weights)):
            content_weight = self.content_weights[i]
            style_weight = self.style_weights[i]
            if content_weight > 0 or style_weight > 0:
                color_target_image = target_image[:, 0:3, :, :]
                color_generated_image = G_output[i][:, 0:3, :, :]
                alpha_target_image = target_image[:, 3, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
                alpha_generated_image = G_output[i][:, 3, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
                image_content_loss = torch.zeros(1, device=self.device)
                image_style_loss = torch.zeros(1, device=self.device)

                for module in self.get_vgg16_modules():
                    color_target_image = module(color_target_image)
                    color_generated_image = module(color_generated_image)
                    alpha_target_image = module(alpha_target_image)
                    alpha_generated_image = module(alpha_generated_image)
                    if content_weight > 0:
                        image_content_loss = image_content_loss \
                                             + (color_target_image - color_generated_image).abs().mean() \
                                             + (alpha_target_image - alpha_generated_image).abs().mean()
                    if style_weight > 0:
                        color_target_gram = compute_gram_matrix(color_target_image)
                        color_generated_gram = compute_gram_matrix(color_generated_image)
                        alpha_target_gram = compute_gram_matrix(alpha_target_image)
                        alpha_generated_gram = compute_gram_matrix(alpha_generated_image)
                        image_style_loss = image_style_loss \
                                           + (color_target_gram - color_generated_gram).abs().mean() \
                                           + (alpha_target_gram - alpha_generated_gram).abs().mean()

                image_content_loss *= content_weight
                image_style_loss *= style_weight
            else:
                image_content_loss = torch.zeros(1, device=self.device)
                image_style_loss = torch.zeros(1, device=self.device)

            content_losses.append(image_content_loss)
            style_losses.append(image_style_loss)

        content_loss = torch.zeros(1, device=self.device)
        for image_content_loss in content_losses:
            content_loss += image_content_loss
        for image_style_loss in style_losses:
            content_loss += image_style_loss

        if log_func is not None:
            for i in range(len(content_losses)):
                log_func("content_loss_%03d" % i, content_losses[i])
            for i in range(len(style_losses)):
                log_func("style_loss_%03d" % i, style_losses[i])
            log_func("content_loss", content_loss)

        return content_loss
