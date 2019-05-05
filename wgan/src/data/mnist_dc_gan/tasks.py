import torch

from pytasuku import Workspace
from wgangp.ls_loss import LsLoss
from wgangp.mnist_dcgan import MnistDcGan
from wgangp.mnist_gan_tasks import MnistGanTasks


def define_tasks(workspace: Workspace):
    cuda = torch.device('cuda')
    MnistGanTasks(workspace,
                  dir="data/mnist_dc_gan",
                  gan_spec=MnistDcGan(device=cuda),
                  loss_spec=LsLoss(device=cuda),
                  learning_rate=1e-3,
                  discriminator_iter_per_generator_iter=1,
                  epoch_per_save_point=1,
                  save_point_count=10)\
        .define_tasks()