import torch

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, Tanh, Sigmoid, Linear

from hana.rindou.nn.image_discriminator_body import ImageDiscriminatorBody
from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec
from hana.rindou.poser.v1.simplified_pumarola import SimplifiedPumarolaDiscriminatorJoined, \
    SimplifiedPumarolaDiscriminatorSeparate
from hana.rindou.nn.image_generator_bodies import bottleneck_generator_body, UNetGeneratorBody
from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class Pumarola(PoserGanSpec):
    def __init__(self, image_size=256, pose_size=12, bone_parameter_count=3,
                 initialization_method='he',
                 insert_residual_blocks=True,
                 discriminator_mode='joined'):
        self._image_size = image_size
        self._pose_size = pose_size
        self._bone_parameter_count = bone_parameter_count
        self.initialization_method = initialization_method
        self.insert_residual_blocks = insert_residual_blocks
        self.discriminator_mode = discriminator_mode

        if discriminator_mode not in ['joined', 'separate', 'reality_score_only', 'do_nothing']:
            raise RuntimeError("discriminator_mode must be one of "
                               "'joined', 'separate', 'reality_score_only' or 'do_nothing'")

    def requires_discriminator_optimization(self) -> bool:
        return self.discriminator_mode != 'do_nothing'

    def image_size(self) -> int:
        return self._image_size

    def pose_size(self) -> int:
        return self._pose_size

    def bone_parameter_count(self) -> int:
        return self._bone_parameter_count

    def generator(self) -> Module:
        return PumarolaGenerator(self._image_size, self._pose_size,
                                 initialization_method=self.initialization_method)

    def discriminator(self) -> Module:
        if self.discriminator_mode == 'joined':
            return SimplifiedPumarolaDiscriminatorSeparate(self._image_size, self._pose_size,
                                                           initialization_method=self.initialization_method,
                                                           insert_residual_blocks=self.insert_residual_blocks)
        elif self.discriminator_mode == 'separate':
            return SimplifiedPumarolaDiscriminatorJoined(self._image_size, self._pose_size,
                                                         initialization_method=self.initialization_method,
                                                         insert_residual_blocks=self.insert_residual_blocks)
        elif self.discriminator_mode == 'reality_score_only':
            return PumarolaGanDiscriminatorNoPoseRegression(
                image_size=self._image_size,
                initialization_method=self.initialization_method,
                insert_residual_blocks=self.insert_residual_blocks)
        elif self.discriminator_mode == 'do_nothing':
            return DiscriminatorDoNothing()
        else:
            raise RuntimeError("discriminator_mode must be one of "
                               "'joined', 'separate', 'reality_score_only' or 'do_nothing'")


class PumarolaGenerator(PoserGanModule):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 body_type='bottleneck'):
        super().__init__()
        self._image_size = image_size
        self._pose_isze = pose_size
        self.intial_dim = initial_dim
        self.bottleneck_image_size = bottleneck_image_size

        if body_type == "bottleneck":
            main_module = bottleneck_generator_body(
                image_size, 4 + pose_size, initial_dim,
                bottleneck_image_size, bottleneck_block_count,
                initialization_method)
        elif body_type == "unet":
            main_module = UNetGeneratorBody(
                image_size, 4 + pose_size, initial_dim,
                bottleneck_image_size, bottleneck_block_count,
                initialization_method)
        else:
            raise RuntimeError("Invalid body_type: %s" % body_type)
        self.main = main_module

        init = create_init_function(initialization_method)
        self.color_change = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Tanh())
        self.alpha_mask = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Sigmoid())

    def forward(self, image: Tensor, pose: Tensor):
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))
        x = torch.cat([image, pose], dim=1)
        y = self.main(x)
        color = self.color_change(y)
        alpha = self.alpha_mask(y)
        output_image = alpha * image + (1 - alpha) * color
        return [output_image, alpha, color]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])


class PumarolaGanDiscriminatorNoPoseRegression(Module):
    def __init__(self,
                 image_size=256,
                 initial_dim=16,
                 max_dim=1024,
                 initialization_method='he',
                 insert_residual_blocks=True):
        super().__init__()
        init = create_init_function(initialization_method)
        self.main = ImageDiscriminatorBody(
            image_size, initial_dim, max_dim, initialization_method, insert_residual_blocks)
        self.reality_score = init(Conv2d(self.main.out_dim, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, input_image, output_image):
        x = torch.cat([input_image, output_image], dim=1)
        y = self.main(x)
        return self.reality_score(y).squeeze()


class DiscriminatorDoNothing(PoserGanModule):
    def __init__(self):
        super().__init__()
        self.module = Linear(in_features=1, out_features=1)

    def forward(self, input_image, *args):
        n = input_image.shape[0]
        return torch.zeros(n, device=input_image.device)

    def forward_from_batch(self, batch):
        return self.forward(*batch)


if __name__ == "__main__":
    generator = PumarolaGenerator()
    state_dict = generator.state_dict()
    for key in state_dict:
        #print(key)
        print("\"%s\"," % key)