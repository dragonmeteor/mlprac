import torch

from data.anime_face.data_loader import anime_face_data_loader
from gans.simplified_style_gan import SimplifiedStyleGan
from gans.style_gan_tasks import StyleGanTasks
from gans.zero_gp_loss import ZeroGpLoss
from pytasuku import Workspace


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')
    StyleGanTasks(
        workspace=workspace,
        prefix="data/anime_face_style_gan",
        output_image_size=64,
        style_gan_spec=SimplifiedStyleGan(),
        loss_spec=ZeroGpLoss(grad_loss_weight=100.0, device=cuda),
        data_loader_func=anime_face_data_loader,
        device=cuda,
        generator_module_learning_rate=1e-4,
        discriminator_learning_rate=1e-4,
        mapping_module_learning_rate=1e-6,
        generator_module_betas=(0.5, 0.9),
        mapping_module_betas=(0.5, 0.9),
        discriminator_betas=(0.5, 0.9))
