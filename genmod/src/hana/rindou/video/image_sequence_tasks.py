import abc
import os
from abc import ABC
from typing import List, Iterable

from pytasuku import Workspace
from pytasuku.indexed.bundled_indexed_file_tasks import BundledIndexedFileTasks
from pytasuku.indexed.indexed_file_tasks import IndexedFileTasks
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks


class ImageSequenceTasks(BundledIndexedFileTasks, ABC):
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 frame_count: int,
                 render_dependencies: List[str] = None):
        self.render_dependencies = render_dependencies
        self.frame_count = frame_count
        self.prefix = prefix
        self.workspace = workspace

        self.render_done_tasks = RenderDoneTasks(self)
        self.indexed_file_tasks = {
            "render_done": self.render_done_tasks
        }
        self.render_done_tasks.define_tasks()

    @property
    def indexed_file_tasks_command_names(self) -> Iterable[str]:
        return self.indexed_file_tasks.keys()

    def get_indexed_file_tasks(self, command_name) -> IndexedFileTasks:
        return self.indexed_file_tasks[command_name]

    def frame_file_name(self, frame_index):
        return self.prefix + ("/frames/%08d.png" % frame_index)

    @abc.abstractmethod
    def render_frame(self, frame_index):
        pass

    def render(self):
        for frame_index in range(self.frame_count):
            self.render_frame(frame_index)
            print("Done with %s ..." % self.frame_file_name(frame_index))
        os.makedirs(os.path.dirname(self.render_done_tasks.file_name), exist_ok=True)
        with open(self.render_done_tasks.file_name, "wt") as fout:
            fout.write("DONE!!!\n")


class RenderDoneTasks(NoIndexFileTasks):
    def __init__(self, image_seq_tasks: ImageSequenceTasks):
        super().__init__(
            image_seq_tasks.workspace,
            image_seq_tasks.prefix,
            "render_done",
            False)
        self.image_seq_tasks = image_seq_tasks

    @property
    def file_name(self):
        return self.prefix + "/render_done.txt"

    def create_file_task(self):
        dependencies = []
        if self.image_seq_tasks.render_dependencies is not None:
            dependencies = self.image_seq_tasks.render_dependencies
        self.workspace.create_file_task(
            self.file_name,
            dependencies,
            self.image_seq_tasks.render)
