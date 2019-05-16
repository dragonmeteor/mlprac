import torch

from data.anime_face.data_loader import anime_face_data_loader
from data.anime_face_pggan.interpolation_video_tasks import InterpolationVideoTasks
from data.anime_face_pggan.training_video_tasks import TrainingVideoTasks
from gans.pggan_tasks import PgGanTasks
from gans.ra_sgan_loss import RaSGanLoss
from gans.zero_gp_loss import ZeroGpLoss
from pytasuku import Workspace


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')
    pg_gan_tasks = PgGanTasks(
        workspace=workspace,
        dir="data/anime_face_pggan",
        output_image_size=64,
        loss_spec=ZeroGpLoss(grad_loss_weight=100.0, device=cuda),
        data_loader_func=anime_face_data_loader,
        device=cuda,
        generator_learning_rate=1e-4,
        discriminator_learning_rate=3e-4,
        generator_betas=(0.5, 0.9),
        discriminator_betas=(0.5, 0.9))
    pg_gan_tasks.define_tasks()

    TrainingVideoTasks(workspace, pg_gan_tasks.dir + "/training_video", pg_gan_tasks).define_tasks()

    InterpolationVideoTasks(workspace, pg_gan_tasks.dir + "/interpolation", pg_gan_tasks).define_tasks()