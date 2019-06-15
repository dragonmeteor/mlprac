import abc
from typing import Tuple, List

from pytasuku import Workspace


class IndexedFileTasks:
    __metaclass__ = abc.ABC

    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @property
    @abc.abstractmethod
    def run_command(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def clean_command(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def shape(self) -> List[int]:
        pass

    @property
    @abc.abstractmethod
    def arity(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def file_list(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get_file_name(self, *indices: int) -> str:
        pass


    @abc.abstractmethod
    def define_tasks(self):
        pass
