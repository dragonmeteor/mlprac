import torch
from torch.nn import Module, Conv2d

from hana.rindou.nn.image_generator_bodies import UNetGeneratorBody
from hana.rindou.nn2.init_function import create_init_function
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class EyeLocRegressor(PoserGanModule):
    def __init__(self,
                 image_size: int = 256,
                 image_dim: int = 4,
                 initialization_method: str = 'he'):
        super().__init__()
        self.initialization_method = initialization_method
        self.image_dim = image_dim
        self.image_size = image_size

        init = create_init_function(initialization_method)

        self.body = UNetGeneratorBody(image_size, image_dim, 16, 32, 6, initialization_method)
        self.to_score = init(Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))

    def forward(self, image):
        score = self.to_score(self.body(image))
        n = image.shape[0]

        coords = torch.linspace(0.5, self.image_size - 0.5, self.image_size, device=image.device) / self.image_size
        left_score = (score[:, 0, :, :].mean(dim=2) * coords.unsqueeze(0).repeat([n, 1])).mean(dim=1)
        right_score = (score[:, 1, :, :].mean(dim=2) * coords.unsqueeze(0).repeat([n, 1])).mean(dim=1)
        bottom_score = (score[:, 2, :, :].mean(dim=1) * coords.unsqueeze(0).repeat([n, 1])).mean(dim=1)
        top_score = (score[:, 3, :, :].mean(dim=1) * coords.unsqueeze(0).repeat([n, 1])).mean(dim=1)

        output = torch.cat([
            left_score.unsqueeze(dim=1),
            right_score.unsqueeze(dim=1),
            1.0 - bottom_score.unsqueeze(dim=1),
            1.0 - top_score.unsqueeze(dim=1)],
            dim=1)
        return [output]

    def forward_from_batch(self, batch):
        return self.forward(batch[0])


class EyeLocRegressorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256,
                 image_dim: int = 4,
                 initialization_method: str = 'he'):
        self.initialization_method = initialization_method
        self.image_dim = image_dim
        self.image_size = image_size

    def get_module(self) -> PoserGanModule:
        return EyeLocRegressor(self.image_size, self.image_dim, self.initialization_method)

    def requires_optimization(self) -> bool:
        return True


if __name__ == "__main__":
    cuda = torch.device("cuda")
    regressor = EyeLocRegressor().to(cuda)

    image = torch.zeros(8, 4, 256, 256, device=cuda)
    regressor(image)
