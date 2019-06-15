import torch

from data.anime_face.data_loader import anime_face_data_loader
from data.anime_face_style_gan.interpolation_video_tasks import InterpolationVideoTasks
from gans.karras_2018_style_gan import Karras2018StyleGan
from gans.simplified_style_gan import SimplifiedStyleGan
from gans.style_gan_tasks import StyleGanTasks
from gans.wgan_gp_loss import WganGpWithDriftLoss
from gans.zero_gp_loss import ZeroGpLoss
from pytasuku import Workspace


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')
    style_gan_tasks = StyleGanTasks(
        workspace=workspace,
        prefix="data/anime_face_style_gan",
        output_image_size=64,
        #style_gan_spec=SimplifiedStyleGan(),
        style_gan_spec=Karras2018StyleGan(),
        loss_spec=WganGpWithDriftLoss(grad_loss_weight=10.0, device=cuda),
        data_loader_func=anime_face_data_loader,
        device=cuda,
        #generator_module_learning_rate=1e-4,
        #discriminator_learning_rate=3e-4,
        #mapping_module_learning_rate=5e-6,
        generator_module_learning_rate=1e-3,
        discriminator_learning_rate=1e-3,
        mapping_module_learning_rate=1e-5,
        generator_module_betas=(0.0, 0.99),
        mapping_module_betas=(0.0, 0.99),
        discriminator_betas=(0.0, 0.99))
    InterpolationVideoTasks(workspace, "data/anime_face_style_gan/interpolation_video", style_gan_tasks).define_tasks()
