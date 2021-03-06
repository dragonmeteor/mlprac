import torch
from torch.nn import Conv2d, LeakyReLU, Sequential, InstanceNorm2d

from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class PumarolaPatchGanTwoOutpusDiscriminator(PoserGanModule):
    """
    A GAN discriminator that accepts (1) the source image, (2) the pose, and (3) two output images.
    """
    def __init__(self,
                 image_dim: int = 4,
                 pose_dim: int = 3,
                 initial_dim: int = 64,
                 repeat_num=6,
                 initialization_method: str = 'he'):
        super().__init__()
        init = create_init_function(initialization_method)

        layers = []
        layers.append(init(Conv2d(3 * image_dim + pose_dim, initial_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(InstanceNorm2d(num_features=initial_dim))
        layers.append(LeakyReLU(0.01, inplace=True))

        curr_dim = initial_dim
        for i in range(1, repeat_num):
            layers.append(init(Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            layers.append(InstanceNorm2d(num_features=curr_dim * 2))
            layers.append(LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        layers.append(init(Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)))
        self.main = Sequential(*layers)

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1], batch[2], batch[3])

    def forward(self, source_image, pose, target_image_0, target_image_1):
        h = source_image.size(2)
        w = source_image.size(3)
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), h, w)
        x = torch.cat([source_image, pose, target_image_0, target_image_1], dim=1)
        output = self.main(x)
        return [output]


class PumarolaPatchGanTwoOutputsDiscriminatorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_dim: int = 4,
                 pose_dim: int = 3,
                 initial_dim: int = 64,
                 repeat_num=6,
                 initialization_method: str = 'he'):
        super().__init__()
        self.repeat_num = repeat_num
        self.pose_dim = pose_dim
        self.initial_dim = initial_dim
        self.image_dim = image_dim
        self.initialization_method = initialization_method

    def requires_optimization(self) -> bool:
        return True

    def get_module(self) -> PoserGanModule:
        return PumarolaPatchGanTwoOutpusDiscriminator(self.image_dim, self.pose_dim, self.initial_dim, self.repeat_num,
                                                      self.initialization_method)


if __name__ == "__main__":
    cuda = torch.device("cuda")

    D = PumarolaPatchGanTwoOutpusDiscriminator().to(cuda)
    source_image = torch.zeros(8, 4, 256, 256).to(cuda)
    target_image_0 = torch.zeros(8, 4, 256, 256).to(cuda)
    target_image_1 = torch.zeros(8, 4, 256, 256).to(cuda)
    pose = torch.zeros(8, 3, device=cuda)
    output = D(source_image, pose, target_image_0, target_image_1)[0]
    print(output.shape)
