from pytasuku import Workspace
import data.mnist.tasks
import data.mnist_dc_gan.tasks
import data.mnist_dc_wgan.tasks


def define_tasks(workspace: Workspace):
    data.mnist.tasks.define_tasks(workspace)
    data.mnist_dc_gan.tasks.define_tasks(workspace)
    data.mnist_dc_wgan.tasks.define_tasks(workspace)