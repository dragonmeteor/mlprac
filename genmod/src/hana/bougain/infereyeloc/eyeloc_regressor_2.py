import torch
from torch.nn import Module, Conv2d, ReLU, Sequential

from hana.rindou.nn2.conv import Conv7Block
from hana.rindou.nn2.init_function import create_init_function
from hana.rindou.nn2.resnet_block import ResNetBlock
from hana.rindou.nn2.view_change import ViewImageAsVector
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class EyeLocRegressor2(Module):
    def __init__(self,
                 image_size: int = 256,
                 image_dim: int = 4,
                 initial_dim: int = 64,
                 initialization_method: str = 'he'):
        super().__init__()
        self.image_size = image_size
        init = create_init_function(initialization_method)

        modules = []
        modules.append(Conv7Block(image_dim, initial_dim, initialization_method))

        num_channels = initial_dim
        while image_size > 1:
            modules.append(ResNetBlock(num_channels, initialization_method))
            num_new_channels = min(512, num_channels * 2)
            new_size = image_size // 2
            modules.append(init(Conv2d(num_channels, num_new_channels, kernel_size=4, stride=2, padding=1, bias=True)))
            modules.append(ReLU(inplace=True))
            image_size = new_size
            num_channels = num_new_channels
        modules.append(Conv2d(num_channels, 4, kernel_size=1, stride=1, padding=0, bias=True))
        modules.append(ViewImageAsVector())
        self.main = Sequential(*modules)

    def forward(self, image):
        return [self.main(image)]

    def forward_from_batch(self, batch):
        return self.forward(batch[0])


class EyeLocRegressor2Spec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256,
                 image_dim: int = 4,
                 initial_dim: int = 64,
                 initialization_method: str = 'he'):
        self.initial_dim = initial_dim
        self.initialization_method = initialization_method
        self.image_dim = image_dim
        self.image_size = image_size

    def get_module(self) -> PoserGanModule:
        return EyeLocRegressor2(self.image_size, self.image_dim, self.initial_dim, self.initialization_method)

    def requires_optimization(self) -> bool:
        return True


if __name__ == "__main__":
    cuda = torch.device("cuda")
    regressor = EyeLocRegressor2().to(cuda)

    image = torch.zeros(8, 4, 256, 256, device=cuda)
    output = regressor(image)
    print(output[0].shape)