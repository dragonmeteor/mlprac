import torch
from cgan.mnist_cgan_tasks import MnistCganTasks
from cgan.mnist_ls_cgan import MnistLsCgan
from pytasuku import Workspace


def define_tasks(workspace: Workspace):
    gan_spec = MnistLsCgan(torch.device('cuda'))
    MnistCganTasks(workspace,
                   "data/mnist_ls_cgan",
                   gan_spec,
                   learning_rate=1e-4,
                   epoch_per_save_point=10).define_tasks()
