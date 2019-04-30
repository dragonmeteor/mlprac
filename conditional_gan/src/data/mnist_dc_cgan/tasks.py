import torch
from cgan.mnist_cgan_tasks import MnistCganTasks
from cgan.mnist_dc_cgan import MnistDcCgan
from pytasuku import Workspace


def define_tasks(workspace: Workspace):
    gan_spec = MnistDcCgan(torch.device('cuda'))
    MnistCganTasks(workspace,
                   "data/mnist_dc_cgan",
                   gan_spec,
                   learning_rate=1e-3,
                   epoch_per_save_point=1).define_tasks()
