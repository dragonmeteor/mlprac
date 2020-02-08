from pytasuku import Workspace
import data._20200206.tasks
import data._20200208.tasks


def define_tasks(workspace: Workspace):
    data._20200206.tasks.define_tasks(workspace)
    data._20200208.tasks.define_tasks(workspace)