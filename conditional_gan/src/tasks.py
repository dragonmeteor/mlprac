from cgan.mnist_cgan_tasks import MnistCganTasks
from pytasuku import Workspace
import data.mnist.tasks


def define_tasks(workspace: Workspace):
    data.mnist.tasks.define_tasks(workspace)
    #MnistCganTasks(workspace, "data/mnist_cgan").define_tasks()