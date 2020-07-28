import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, Tanh

from hana.rindou.nn.image_discriminator_body import ImageDiscriminatorBody
from hana.rindou.nn.image_generator_bodies import bottleneck_generator_body
from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec


class SimplifiedPumarola(PoserGanSpec):
    def __init__(self, image_size=256, pose_size=12, bone_parameter_count=3,
                 initialization_method='he',
                 insert_residual_blocks=True,
                 use_separate_discriminator=False):
        self._image_size = image_size
        self._pose_size = pose_size
        self._bone_parameter_count = bone_parameter_count
        self.initialization_method = initialization_method
        self.insert_residual_blocks = insert_residual_blocks
        self.use_separate_discriminator = use_separate_discriminator

    def requires_discriminator_optimization(self) -> bool:
        return True

    def pose_size(self):
        return self._pose_size

    def image_size(self):
        return self._image_size

    def bone_parameter_count(self):
        return self._bone_parameter_count

    def generator(self) -> Module:
        return SimplifiedPumarolaGenerator(self._image_size, self._pose_size,
                                           initialization_method=self.initialization_method)

    def discriminator(self) -> Module:
        if self.use_separate_discriminator:
            return SimplifiedPumarolaDiscriminatorSeparate(self._image_size, self._pose_size,
                                                           initialization_method=self.initialization_method,
                                                           insert_residual_blocks=self.insert_residual_blocks)
        else:
            return SimplifiedPumarolaDiscriminatorJoined(self._image_size, self._pose_size,
                                                         initialization_method=self.initialization_method,
                                                         insert_residual_blocks=self.insert_residual_blocks)


# Copied from https://github.com/albertpumarola/GANimation/blob/master/networks/generator_wasserstein_gan.py
class SimplifiedPumarolaGenerator(Module):
    def __init__(self, image_size=256, pose_size=12,
                 initial_dim=64, bottleneck_image_size=32, bottleneck_block_count=6,
                 initialization_method='he'):
        super().__init__()
        self._image_size = image_size
        self._pose_size = pose_size
        self.initial_dim = initial_dim
        self.bottleneck_image_size = bottleneck_image_size

        main_module = bottleneck_generator_body(
            image_size, 4 + pose_size, initial_dim,
            bottleneck_image_size, bottleneck_block_count,
            initialization_method)
        self.main = main_module

        init = create_init_function(initialization_method)
        self.image = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Tanh())

    def forward(self, image: Tensor, pose: Tensor):
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))
        x = torch.cat([image, pose], dim=1)
        return self.image(self.main(x)),


class SimplifiedPumarolaDiscriminatorJoined(Module):
    def __init__(self, image_size=256, pose_size=12, initial_dim=16, max_dim=1024,
                 initialization_method='he',
                 insert_residual_blocks=True):
        super().__init__()
        init = create_init_function(initialization_method)
        self.main = ImageDiscriminatorBody(
            image_size, initial_dim, max_dim, initialization_method, insert_residual_blocks)
        self.reality_score = init(Conv2d(self.main.out_dim, 1, kernel_size=1, stride=1, padding=0, bias=False))
        self.pose = init(Conv2d(self.main.out_dim, pose_size, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, input_image, output_image):
        x = torch.cat([input_image, output_image], dim=1)
        y = self.main(x)
        return self.reality_score(y).squeeze(), self.pose(y).squeeze()


class SimplifiedPumarolaDiscriminatorSeparate(Module):
    def __init__(self, image_size=256, pose_size=12, initial_dim=16, max_dim=1024,
                 initialization_method='he',
                 insert_residual_blocks=False):
        super().__init__()
        init = create_init_function(initialization_method)
        reality_score_main = ImageDiscriminatorBody(
            image_size, initial_dim, max_dim, initialization_method, insert_residual_blocks)
        pose_main = ImageDiscriminatorBody(
            image_size, initial_dim, max_dim, initialization_method, insert_residual_blocks)
        self.reality_score = Sequential(
            reality_score_main,
            init(Conv2d(self.main.out_dim, 1, kernel_size=1, stride=1, padding=0, bias=False)))
        self.pose = Sequential(
            pose_main,
            init(Conv2d(self.main.out_dim, pose_size, kernel_size=1, stride=1, padding=0, bias=False)))

    def forward(self, input_image, output_image):
        x = torch.cat([input_image, output_image], dim=1)
        return self.reality_score(x).squeeze(), self.pose(x).squeeze()


if __name__ == "__main__":
    spec = SimplifiedPumarola()

    cuda = torch.device('cuda')
    G = spec.generator().to(cuda)
    D = spec.discriminator().to(cuda)

    batch_size = 32
    source_image = torch.zeros(batch_size, 4, 256, 256, device=cuda)
    target_image = torch.zeros(batch_size, 4, 256, 256, device=cuda)
    pose = torch.zeros(batch_size, 12, device=cuda)

    G_output = G(source_image, pose)
    print(G_output.shape)

    score, pose = D(source_image, target_image)
    print(score.shape)
    print(pose.shape)
