import os
from typing import Iterable, Callable, List

from hana.rindou.util import convert_avs_to_avi, convert_avi_to_webm, convert_avi_to_mp4
from pytasuku import Workspace
from pytasuku.indexed.bundled_indexed_file_tasks import BundledIndexedFileTasks
from pytasuku.indexed.indexed_file_tasks import IndexedFileTasks
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks


class ImageSequenceVideoTasks(BundledIndexedFileTasks):
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 frame_count: int,
                 frame_file_name_func: Callable[[int], str],
                 fps: float,
                 frame_dependencies: List[str] = None,
                 define_tasks_immediately: bool = True):
        super().__init__()
        self.frame_dependencies = frame_dependencies
        self.fps = fps
        self.frame_file_name_func = frame_file_name_func
        self.frame_count = frame_count
        self.prefix = prefix
        self.workspace = workspace

        self.avs_tasks = AvsTasks(self)
        self.avi_tasks = AviTasks(self)
        self.mp4_tasks = Mp4Tasks(self)
        self.webm_tasks = WebmTasks(self)
        self.indexed_file_tasks = {
            "avs": self.avs_tasks,
            "avi": self.avi_tasks,
            "mp4": self.mp4_tasks,
            "webm": self.webm_tasks,
        }

        if define_tasks_immediately:
            for tasks in self.indexed_file_tasks.values():
                tasks.define_tasks()

    @property
    def indexed_file_tasks_command_names(self) -> Iterable[str]:
        return self.indexed_file_tasks.keys()

    def get_indexed_file_tasks(self, command_name) -> IndexedFileTasks:
        return self.indexed_file_tasks[command_name]

    def create_avs_file(self):
        os.makedirs(os.path.dirname(self.avs_tasks.file_name), exist_ok=True)
        with open(self.avs_tasks.file_name, "w") as fout:
            n = self.frame_count
            for i in range(n):
                frame_path = os.path.relpath(self.frame_file_name_func(i), self.prefix)
                frame_clip = "ImageSource(\"%s\", start = 0, end = 0, fps = %f)" % (frame_path, self.fps)
                fout.write(frame_clip)
                if i < n - 1:
                    fout.write(" + \\\n")

    def create_avi_file(self):
        convert_avs_to_avi(self.avs_tasks.file_name, self.avi_tasks.file_name)

    def create_mp4_file(self):
        convert_avi_to_mp4(self.avi_tasks.file_name, self.mp4_tasks.file_name)

    def create_webm_file(self):
        convert_avi_to_webm(self.avi_tasks.file_name, self.webm_tasks.file_name)


class AvsTasks(NoIndexFileTasks):
    def __init__(self, video_tasks: ImageSequenceVideoTasks):
        super().__init__(
            video_tasks.workspace,
            video_tasks.prefix,
            "avs",
            False)
        self.video_tasks = video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.avs"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [],
            self.video_tasks.create_avs_file)


class AviTasks(NoIndexFileTasks):
    def __init__(self, video_tasks: ImageSequenceVideoTasks):
        super().__init__(
            video_tasks.workspace,
            video_tasks.prefix,
            "avi",
            False)
        self.video_tasks = video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.avi"

    def create_file_task(self):
        dependencies = [self.video_tasks.avs_tasks.file_name]
        if self.video_tasks.frame_dependencies is not None:
            dependencies = dependencies + self.video_tasks.frame_dependencies
        self.workspace.create_file_task(
            self.file_name,
            dependencies,
            self.video_tasks.create_avi_file)


class Mp4Tasks(NoIndexFileTasks):
    def __init__(self, video_tasks: ImageSequenceVideoTasks):
        super().__init__(
            video_tasks.workspace,
            video_tasks.prefix,
            "mp4",
            False)
        self.video_tasks = video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.mp4"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [
                self.video_tasks.avi_tasks.file_name,
            ],
            self.video_tasks.create_mp4_file)


class WebmTasks(NoIndexFileTasks):
    def __init__(self, video_tasks: ImageSequenceVideoTasks):
        super().__init__(
            video_tasks.workspace,
            video_tasks.prefix,
            "webm",
            False)
        self.video_tasks = video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.webm"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [
                self.video_tasks.avi_tasks.file_name,
            ],
            self.video_tasks.create_webm_file)
