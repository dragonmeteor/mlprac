import torch

from data.anime_face.data_loader import anime_face_data_loader
from gans.pggan_tasks import PgGanTasks
from gans.wgan_gp_loss import WganGpLoss
from pytasuku import Workspace


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')
    tasks = PgGanTasks(
        workspace=workspace,
        dir="data/anime_face_pggan",
        image_size=64,
        loss_spec=WganGpLoss(grad_loss_weight=10.0, device=cuda),
        data_loader_func=anime_face_data_loader,
        device=cuda)
    tasks.define_tasks()
