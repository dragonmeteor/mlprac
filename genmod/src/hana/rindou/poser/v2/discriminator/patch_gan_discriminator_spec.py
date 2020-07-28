import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, InstanceNorm2d

from hana.rindou.nn.init_function import create_init_function
from hana.rindou.nn.residual_block import ResidualBlock
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class PatchGanDiscriminator(PoserGanModule):
    def __init__(self,
                 image_dim: int = 4,
                 pose_size: int = 3,
                 initial_dim: int = 32,
                 max_dim: int = 1024,
                 down_sample_count: int = 4,
                 insert_residual_block: bool = True,
                 initialization_method: str = 'he'):
        super().__init__()
        init = create_init_function(initialization_method)
        layers = []

        layers.append(
            init(Conv2d(
                in_channels=2 * image_dim + pose_size,
                out_channels=initial_dim,
                kernel_size=7,
                padding=3,
                stride=1,
                bias=True)))
        layers.append(InstanceNorm2d(num_features=initial_dim))
        layers.append(ReLU())

        current_dim = initial_dim
        for i in range(down_sample_count):
            new_dim = min(max_dim, current_dim * 2)
            layers.append(
                init(Conv2d(in_channels=current_dim, out_channels=new_dim, kernel_size=4, stride=2, padding=1)))
            layers.append(InstanceNorm2d(num_features=new_dim))
            layers.append(ReLU())
            current_dim = new_dim
            if insert_residual_block:
                layers.append(ResidualBlock(dim=current_dim, initialization_method=initialization_method))

        layers.append(Conv2d(in_channels=current_dim, out_channels=1, kernel_size=1, stride=1, padding=0))

        self.body = Sequential(*layers)

    def forward(self, source_image, pose, generated_image):
        h = source_image.size(2)
        w = source_image.size(3)
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), h, w)

        x = torch.cat([source_image, pose, generated_image], dim=1)
        return [self.body(x)]

    def forward_from_batch(self, batch):
        self.forward(batch[0], batch[1], batch[2])



class PatchGanDiscriminatorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_dim: int = 4,
                 pose_size: int = 3,
                 initial_dim: int = 32,
                 max_dim: int = 1024,
                 down_sample_count: int = 4,
                 insert_residual_block: bool = True,
                 initialization_method: str = 'he',
                 requires_optimization: bool = True):
        self._image_dim = image_dim
        self._pose_size = pose_size
        self._initial_dim = initial_dim
        self._max_dim = max_dim
        self._down_sample_count = down_sample_count
        self._insert_residual_block = insert_residual_block
        self._initialization_method = initialization_method
        self._requires_optimization = requires_optimization

    def requires_optimization(self) -> bool:
        return self._requires_optimization

    def get_module(self) -> PoserGanModule:
        return PatchGanDiscriminator(
            self._image_dim,
            self._pose_size,
            self._initial_dim,
            self._max_dim,
            self._down_sample_count,
            self._insert_residual_block,
            self._initialization_method)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    D = PatchGanDiscriminator().to(cuda)
    source_image = torch.zeros(8, 4, 256, 256, device=cuda)
    target_image = torch.zeros(8, 4, 256, 256, device=cuda)
    pose = torch.zeros(8, 3, device=cuda)
    output = D(source_image, pose, target_image)
    print(output[0].shape)
