from typing import List, Callable

import torch
from torch import Tensor
from torch.nn import Module
from torchvision.models import vgg16

from hana.rindou.nn.vgg16 import get_vgg16_perceptual_loss_modules
from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss


class PixelAndPerceptualLoss(PoserGanLoss):
    def __init__(self,
                 layers_used: List[str],
                 pixel_loss_weight: float,
                 content_loss_weight: float,
                 device: torch.device = torch.device('cpu')):
        self.pixel_loss_weight = pixel_loss_weight
        self.content_loss_weight = content_loss_weight
        self.layers_used = layers_used
        self.device = device
        self.vgg16_modules = None

    def get_vgg16_modules(self):
        if self.vgg16_modules is None:
            self.vgg16_modules = [module.to(self.device)
                                  for module in get_vgg16_perceptual_loss_modules(self.layers_used)]
        return self.vgg16_modules

    def discriminator_loss(self, G: Module, D: Module, batch: List[Tensor],
                           log_func: Callable[[str, float], None] = None) -> Tensor:
        return torch.zeros(1, device=self.device)

    def generator_loss(self, G: Module, D: Module, batch: List[Tensor],
                       log_func: Callable[[str, float], None] = None) -> Tensor:
        source_image = batch[0]
        pose = batch[1]
        target_image = batch[2]
        generated_image = G(source_image, pose)[0]

        if self.pixel_loss_weight > 0:
            pixel_loss = (generated_image - target_image).abs().mean()
        else:
            pixel_loss = torch.zeros(1, device=self.device)

        if self.content_loss_weight > 0:
            color_target_image = target_image[:,0:3,:,:]
            color_generated_image = generated_image[:,0:3,:,:]
            alpha_target_image = target_image[:,3,:,:].unsqueeze(1).expand(-1, 3, -1, -1)
            alpha_generated_image = generated_image[:, 3, :, :].unsqueeze(1).expand(-1, 3, -1, -1)
            content_loss = torch.zeros(1, device=self.device)
            for module in self.get_vgg16_modules():
                color_target_image = module(color_target_image)
                color_generated_image = module(color_generated_image)
                alpha_target_image = module(alpha_target_image)
                alpha_generated_image = module(alpha_generated_image)
                content_loss = content_loss \
                               + (color_target_image - color_generated_image).abs().mean() \
                               + (alpha_target_image - alpha_generated_image).abs().mean()
        else:
            content_loss = torch.zeros(1, device=self.device)

        pixel_loss = self.pixel_loss_weight * pixel_loss
        content_loss = self.content_loss_weight * content_loss
        loss = pixel_loss + content_loss

        if log_func is not None:
            log_func("generator_pixel_loss", pixel_loss.item())
            log_func("generator_content_loss", content_loss.item())
            log_func("generator_loss", loss.item())

        return loss

if __name__ == "__main__":
    model = vgg16(pretrained=True)
    i = 0
    for x in model.features:
        print(i, x)
        i += 1
