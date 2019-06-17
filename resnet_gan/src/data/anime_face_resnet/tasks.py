from typing import List, Tuple

import torch

from data.anime_face.data_loader import anime_face_data_loader
from gans.gan_tasks import GanTasks, GanTrainingSpec, SampleImageSpec
from gans.resnet_64_gan import Resnet64Gan
from gans.wgan_gp_loss import WganGpWithDriftLoss
from gans.wgan_loss import WganLoss
from pytasuku import Workspace


class Resnet64GanTrainingSpec(GanTrainingSpec):
    def __init__(self):
        super().__init__()

    @property
    def save_point_count(self) -> int:
        return 12

    @property
    def sample_per_save_point(self) -> int:
        return 100000

    @property
    def generator_learning_rates(self) -> List[float]:
        return [1e-4 for i in range(self.save_point_count//2)] + [3e-5 for i in range(self.save_point_count//2)]

    @property
    def discriminator_learning_rates(self) -> List[float]:
        return [3e-4 for i in range(self.save_point_count//2)] + [1e-4 for i in range(self.save_point_count//2)]

    @property
    def batch_size(self) -> int:
        return 32

    @property
    def random_seed(self) -> int:
        return 4891531

    @property
    def generator_betas(self) -> Tuple[float, float]:
        return 0.0, 0.9

    @property
    def discriminator_betas(self) -> Tuple[float, float]:
        return 0.0, 0.9


class Resnet64GanSampleImageSpec(SampleImageSpec):

    @property
    def count(self) -> int:
        return 64

    @property
    def image_per_row(self) -> int:
        return 8

    @property
    def sample_per_sample_image(self) -> int:
        return 10000

    @property
    def latent_vector_seed(self) -> int:
        return 12465497


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')

    GanTasks(
        workspace=workspace,
        prefix="data/anime_face_resnet_0",
        gan_spec=Resnet64Gan(),
        loss_spec=WganGpWithDriftLoss(grad_loss_weight=10.0, device=cuda),
        training_spec=Resnet64GanTrainingSpec(),
        sample_image_spec=Resnet64GanSampleImageSpec(),
        data_load_func=lambda batch_size, device: anime_face_data_loader(64, batch_size, device),
        device=cuda)

    GanTasks(
        workspace=workspace,
        prefix="data/anime_face_resnet",
        gan_spec=Resnet64Gan(use_spectral_normalization_in_discriminator=True),
        loss_spec=WganLoss(device=cuda),
        training_spec=Resnet64GanTrainingSpec(),
        sample_image_spec=Resnet64GanSampleImageSpec(),
        data_load_func=lambda batch_size, device: anime_face_data_loader(64, batch_size, device),
        device=cuda)