import abc
from typing import Iterable, List

from pytasuku import Workspace
from pytasuku.indexed.indexed_file_tasks import IndexedFileTasks
from pytasuku.workspace import do_nothing


class BundledIndexedFileTasks:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def indexed_file_tasks_command_names(self) -> Iterable[str]:
        pass

    @abc.abstractmethod
    def get_indexed_file_tasks(self, command_name) -> IndexedFileTasks:
        pass


def define_forall_tasks_from_list(workspace: Workspace, prefix: str, tasks: List[BundledIndexedFileTasks]):
    for command_name in tasks[0].indexed_file_tasks_command_names:
        workspace.create_command_task(
            prefix + "/" + command_name,
            [x.get_indexed_file_tasks(command_name).run_command for x in tasks],
            do_nothing)
        workspace.create_command_task(
            prefix + "/" + command_name + "_clean",
            [x.get_indexed_file_tasks(command_name).clean_command for x in tasks],
            do_nothing)