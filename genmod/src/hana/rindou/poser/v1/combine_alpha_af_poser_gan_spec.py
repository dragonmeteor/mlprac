import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, Tanh, Sigmoid
import torch.nn.functional as F

from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec
from hana.rindou.poser.v1.pumarola import PumarolaGanDiscriminatorNoPoseRegression, DiscriminatorDoNothing
from hana.rindou.poser.v1.simplified_pumarola import SimplifiedPumarolaDiscriminatorSeparate, \
    SimplifiedPumarolaDiscriminatorJoined
from hana.rindou.nn.image_generator_bodies import bottleneck_generator_body


class CombineAlphaAfPoserGanSpec(PoserGanSpec):
    def __init__(self,
                 image_size=256, pose_size=3, bone_parameter_count=3,
                 initialization_method='he',
                 insert_residual_blocks=True,
                 use_separate_discriminator=False,
                 discriminator_mode='joined'):
        self._image_size = image_size
        self._pose_size = pose_size
        self._bone_parameter_count = bone_parameter_count
        self.initialization_method = initialization_method
        self.insert_residual_blocks = insert_residual_blocks
        self.use_separate_discriminator = use_separate_discriminator
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
        return CombineAlphaAfGenerator(
            image_size=self.image_size(),
            pose_size=self.pose_size(),
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


class CombineAlphaAfGenerator(Module):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 3,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he'):
        super().__init__()
        self._image_size = image_size
        self._pose_isze = pose_size
        self.intial_dim = initial_dim
        self.bottleneck_image_size = bottleneck_image_size

        self.face_rotate = bottleneck_generator_body(
            image_size, 4 + pose_size, initial_dim,
            bottleneck_image_size, bottleneck_block_count,
            initialization_method)

        init = create_init_function(initialization_method)
        self.face_rotate_color_change = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Tanh())
        self.face_rotate_alpha_mask = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Sigmoid())
        self.face_rotate_grid_change = init(Conv2d(initial_dim, 2, kernel_size=7, stride=1, padding=3, bias=False))

        self.combine = bottleneck_generator_body(
            image_size, 8 + pose_size, initial_dim,
            bottleneck_image_size, bottleneck_block_count,
            initialization_method)
        self.combine_alpha_mask = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Sigmoid())

    def forward(self, image: Tensor, pose: Tensor):
        n = image.size(0)
        c = image.size(1)
        h = image.size(2)
        w = image.size(3)

        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))
        x = torch.cat([image, pose], dim=1)
        face_rotated = self.face_rotate(x)

        face_rotated_color = self.face_rotate_color_change(face_rotated)
        face_rotated_alpha = self.face_rotate_alpha_mask(face_rotated)
        face_rotated_by_alpha = face_rotated_alpha * image + (1 - face_rotated_alpha) * face_rotated_color

        grid_change = torch.transpose(self.face_rotate_grid_change(face_rotated).view(n, 2, h * w), 1, 2).view(n, h, w,
                                                                                                               2)
        device = self.face_rotate_grid_change.weight.device
        identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = F.affine_grid(identity, [n, c, h, w])
        grid = base_grid + grid_change
        face_rotated_by_sampling = F.grid_sample(image, grid, mode='bilinear', padding_mode='border')

        z = torch.cat([face_rotated_by_alpha, face_rotated_by_sampling, pose], dim=1)
        combined = self.combine(z)
        combined_alpha = self.combine_alpha_mask(combined)
        combined_image = combined_alpha * face_rotated_alpha + (1 - combined_alpha) * face_rotated_by_sampling

        return combined_image, face_rotated_by_alpha, face_rotated_by_sampling
