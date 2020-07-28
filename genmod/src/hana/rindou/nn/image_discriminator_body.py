from torch.nn import Module, Conv2d, LeakyReLU, Sequential

from hana.rindou.nn.init_function import create_init_function
from hana.rindou.nn.residual_block import ResidualBlock

# Copied from https://github.com/albertpumarola/GANimation/blob/master/networks/discriminator_wasserstein_gan.py
class ImageDiscriminatorBody(Module):
    def __init__(self, image_size=256, initial_dim=16, max_dim=1024,
                 initialization_method='he',
                 insert_residual_blocks=False):
        super().__init__()

        init = create_init_function(initialization_method)

        layers = []
        layers.append(init(Conv2d(4 * 2, initial_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(LeakyReLU(0.01))

        current_size = image_size // 2
        current_dim = initial_dim
        while current_size > 1:
            if insert_residual_blocks:
                layers.append(ResidualBlock(current_dim, initialization_method))
            new_dim = min(max_dim, current_dim * 2)
            layers.append(init(Conv2d(current_dim, new_dim, kernel_size=4, stride=2, padding=1)))
            layers.append(LeakyReLU(0.01))
            current_dim = new_dim
            current_size = current_size // 2

        # Fully connected residual block
        if insert_residual_blocks:
            layers.append(ResidualBlock(current_dim, initialization_method, one_pixel=True))

        self.main = Sequential(*layers)
        self.out_dim = current_dim

    def forward(self, x):
        return self.main(x)