from pytasuku import Workspace
import data.mnist.tasks
import data.mnist_ls_cgan.tasks
import data.mnist_dc_cgan.tasks

def define_tasks(workspace: Workspace):
    data.mnist.tasks.define_tasks(workspace)
    data.mnist_ls_cgan.tasks.define_tasks(workspace)
    data.mnist_dc_cgan.tasks.define_tasks(workspace)