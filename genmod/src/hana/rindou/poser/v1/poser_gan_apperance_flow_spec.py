import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Conv2d

from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec
from hana.rindou.poser.v1.pumarola import PumarolaGanDiscriminatorNoPoseRegression, DiscriminatorDoNothing
from hana.rindou.poser.v1.simplified_pumarola import SimplifiedPumarolaDiscriminatorSeparate, \
    SimplifiedPumarolaDiscriminatorJoined
from hana.rindou.nn.image_generator_bodies import bottleneck_generator_body, UNetGeneratorBody
from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule


class PoserGanApperanceFlowGenerator(PoserGanModule):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 body_type: str = 'bottleneck',
                 align_corners: bool = True):
        super().__init__()
        self._image_size = image_size
        self._pose_isze = pose_size
        self.intial_dim = initial_dim
        self.bottleneck_image_size = bottleneck_image_size
        self.align_corners = align_corners
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
        self.grid_change = init(Conv2d(initial_dim, 2, kernel_size=7, stride=1, padding=3, bias=False))

    def forward(self, image: Tensor, pose: Tensor):
        n = image.size(0)
        c = image.size(1)
        h = image.size(2)
        w = image.size(3)

        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), image.size(2), image.size(3))

        x = torch.cat([image, pose], dim=1)
        y = self.main(x)
        grid_change = torch.transpose(self.grid_change(y).view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        device = self.grid_change.weight.device
        identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = F.affine_grid(identity, [n, c, h, w], align_corners=self.align_corners)
        grid = base_grid + grid_change
        resampled_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border',
                                        align_corners=self.align_corners)

        return [resampled_image, grid_change]

    def forward_from_batch(self, batch):
        image = batch[0]
        pose = batch[1]
        return self.forward(image, pose)


class PoserGanWithAppearanceFlowSpec(PoserGanSpec):
    def __init__(self, image_size=256, pose_size=12, bone_parameter_count=3,
                 initialization_method='he',
                 insert_residual_blocks=True,
                 use_separate_discriminator=False,
                 discriminator_mode='joined'):
        super().__init__()
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
        return PoserGanApperanceFlowGenerator(self._image_size, self._pose_size,
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


if __name__ == "__main__":
    # with open("data/rindou/_20190824/pumarola_01_videos/waifulab_00_orig/waifu_256.png", "rb") as file:
    #    image = extract_image_from_filelike(file).unsqueeze(0)
    # grid = F.affine_grid(torch.Tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0), [1, 4, 256, 256])
    # resampled_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border')
    # print(image.shape)
    # print(resampled_image.shape)
    # numpy_image = rgba_to_numpy_image(resampled_image.squeeze())
    # pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
    # pil_image.save("data/rindou/_20190824/pumarola_01_videos/waifulab_00_orig/waifu_256_resampled.png")

    cuda = torch.device('cuda')
    image = torch.zeros(8, 4, 256, 256, device=cuda)
    pose = torch.zeros(8, 3, device=cuda)
    G = PoserGanApperanceFlowGenerator(pose_size=3).to(cuda)
    output_image, grid_change = G(image, pose)
    print(output_image.shape)
    print(output_image.device)
    print(grid_change.shape)
    print(grid_change.device)
