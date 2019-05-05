import torch

from pytasuku import Workspace
from wgangp.mnist_dcgan import MnistDcGan
from wgangp.mnist_gan_tasks import MnistGanTasks
from wgangp.wgan_gp_loss import WganGpLoss


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')
    MnistGanTasks(workspace,
                  dir="data/mnist_dc_wgan",
                  gan_spec=MnistDcGan(device=cuda),
                  loss_spec=WganGpLoss(grad_loss_weight=10.0, device=cuda),
                  learning_rate=1e-4,
                  generator_betas=(0,0.999),
                  discriminator_betas=(0,0.9999),
                  discriminator_iter_per_generator_iter=5,
                  epoch_per_save_point=5,
                  save_point_count=10)\
        .define_tasks()