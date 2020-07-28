import torch
from torch.nn import Conv2d, InstanceNorm2d, ReLU, ConvTranspose2d, Sequential, Module, ModuleList

from hana.rindou.nn.init_function import create_init_function
from hana.rindou.nn.residual_block import ResidualBlock


def bottleneck_generator_body(image_size: int,
                              image_dim: int,
                              initial_dim: int,
                              bottleneck_image_size: int,
                              bottleneck_block_count: int,
                              initialization_method: str = 'he'):
    init = create_init_function(initialization_method)
    layers = []
    layers.append(init(Conv2d(image_dim, initial_dim, kernel_size=7, stride=1, padding=3, bias=False)))
    layers.append(InstanceNorm2d(initial_dim, affine=True))
    layers.append(ReLU(inplace=True))

    # Downsampling
    current_dim = initial_dim
    current_image_size = image_size
    while current_image_size > bottleneck_image_size:
        layers.append(init(Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)))
        layers.append(InstanceNorm2d(current_dim * 2, affine=True))
        layers.append(ReLU(inplace=True))
        current_dim = current_dim * 2
        current_image_size = current_image_size // 2

    for i in range(bottleneck_block_count):
        layers.append(ResidualBlock(dim=current_dim, initialization_method=initialization_method))

    # Upsampling
    while current_image_size < image_size:
        layers.append(
            init(ConvTranspose2d(current_dim, current_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)))
        layers.append(InstanceNorm2d(current_dim // 2, affine=True))
        layers.append(ReLU(inplace=True))
        current_dim = current_dim // 2
        current_image_size = current_image_size * 2

    return Sequential(*layers)


class UNetGeneratorBody(Module):
    def __init__(self,
                 image_size: int,
                 image_dim: int,
                 initial_dim: int,
                 bottleneck_image_size: int,
                 bottleneck_block_count: int,
                 initialization_method: str = 'he'):
        super().__init__()

        init = create_init_function(initialization_method)

        self.downward_modules = ModuleList()
        self.downward_module_channel_count = {}

        self.downward_modules.append(Sequential(
            init(Conv2d(image_dim, initial_dim, kernel_size=7, stride=1, padding=3, bias=False)),
            InstanceNorm2d(initial_dim, affine=True),
            ReLU(inplace=True)))
        self.downward_module_channel_count[image_size] = initial_dim

        # Downsampling
        current_dim = initial_dim
        current_image_size = image_size
        while current_image_size > bottleneck_image_size:
            self.downward_modules.append(
                Sequential(
                    init(Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)),
                    InstanceNorm2d(current_dim * 2, affine=True),
                    ReLU(inplace=True)))
            current_dim = current_dim * 2
            current_image_size = current_image_size // 2
            self.downward_module_channel_count[current_image_size] = current_dim

        # Bottleneck
        self.bottleneck_modules = ModuleList()
        for i in range(bottleneck_block_count):
            self.bottleneck_modules.append(ResidualBlock(dim=current_dim, initialization_method=initialization_method))

        # Upsampling
        self.upsampling_modules = ModuleList()
        while current_image_size < image_size:
            if current_image_size == bottleneck_image_size:
                input_dim = current_dim
            else:
                input_dim = current_dim + self.downward_module_channel_count[current_image_size]
            self.upsampling_modules.insert(0,
                Sequential(
                    init(ConvTranspose2d(input_dim, current_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)),
                    InstanceNorm2d(current_dim // 2, affine=True),
                    ReLU(inplace=True)))
            current_dim = current_dim // 2
            current_image_size = current_image_size * 2

        self.upsampling_modules.insert(0,
            Sequential(
                init(Conv2d(current_dim + initial_dim, initial_dim, kernel_size=7, stride=1, padding=3, bias=False)),
                InstanceNorm2d(initial_dim, affine=True),
                ReLU(inplace=True)))

    def forward(self, x):
        downward_outputs = []
        for module in self.downward_modules:
            x = module(x)
            downward_outputs.append(x)
        for module in self.bottleneck_modules:
            x = module(x)
        x = self.upsampling_modules[-1](x)
        for i in range(len(self.upsampling_modules)-2,-1,-1):
            y = torch.cat([x, downward_outputs[i]], dim=1)
            x = self.upsampling_modules[i](y)
        return x

if __name__ == "__main__":
    cuda = torch.device('cuda')
    unet = UNetGeneratorBody(
        image_size=256, image_dim=7, initial_dim=32, bottleneck_image_size=32, bottleneck_block_count=6).to(cuda)

    x = torch.zeros(8, 7, 256, 256, device=cuda)
    y = unet(x)
    print(y.shape)