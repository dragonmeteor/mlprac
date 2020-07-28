from torch import Tensor
from torch.nn import Module

from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class CombineTwoImagesGenerator(PoserGanModule):
    def __init__(self,
                 two_images_generator_spec: PoserGanModuleSpec,
                 combine_two_images_module_spec: PoserGanModuleSpec):
        super().__init__()
        self.two_images_generator_spec = two_images_generator_spec
        self.combine_two_images_module_spec = combine_two_images_module_spec
        self.two_images_generator = two_images_generator_spec.get_module()
        self.combine_two_images_module = combine_two_images_module_spec.get_module()

    def forward(self, image: Tensor, pose: Tensor):
        generated_images = self.two_images_generator(image, pose)
        if self.two_images_generator_spec.requires_optimization():
            first_image = generated_images[0]
            second_image = generated_images[1]
        else:
            first_image = generated_images[0].detach()
            second_image = generated_images[1].detach()
        return self.combine_two_images_module(first_image, second_image, pose) + [first_image, second_image]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])



class CombineTwoImagesGeneratorSpec(PoserGanModuleSpec):
    def __init__(self,
                 two_images_generator_spec: PoserGanModuleSpec,
                 combine_two_images_module_spec: PoserGanModuleSpec):
        self.two_image_generator_spec = two_images_generator_spec
        self.combine_two_images_module_spec = combine_two_images_module_spec

    def requires_optimization(self) -> bool:
        return self.two_image_generator_spec.requires_optimization() \
               or self.combine_two_images_module_spec.requires_optimization()

    def get_module(self) -> PoserGanModule:
        return CombineTwoImagesGenerator(self.two_image_generator_spec, self.combine_two_images_module_spec)
