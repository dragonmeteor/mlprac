import os
from typing import List, Tuple, Iterable

import PIL.Image
import numpy
import torch
from torch import Tensor

from hana.rindou.poser.poser.poser import Poser
from hana.rindou.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image, convert_avs_to_avi
from pytasuku import Workspace
from pytasuku.indexed.bundled_indexed_file_tasks import BundledIndexedFileTasks
from pytasuku.indexed.indexed_file_tasks import IndexedFileTasks
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks


class PoserVideoTasksVer2(BundledIndexedFileTasks):
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 rest_image_file_name: str,
                 key_frames: List[Tuple[int, Tensor]],
                 include_last_frame: True,
                 poser: Poser,
                 device: torch.device = torch.device("cpu"),
                 render_dependencies: List[str] = None):
        self.render_dependencies = render_dependencies
        self.device = device
        self.poser = poser
        self.key_frames = key_frames
        self.include_last_frame = include_last_frame
        self.rest_image_file_name = rest_image_file_name
        self.prefix = prefix
        self.workspace = workspace

        assert len(self.key_frames) >= 1
        assert self.key_frames[0][0] == 1

        self.render_done_tasks = RenderDoneTasks(self)
        self.avs_tasks = AvsTasks(self)
        self.avi_tasks = AviTasks(self)
        self.mp4_tasks = Mp4Tasks(self)
        self.webm_tasks = WebmTasks(self)

        self.indexed_file_tasks = {
            "render_done": self.render_done_tasks,
            "avs": self.avs_tasks,
            "avi": self.avi_tasks,
            "mp4": self.mp4_tasks,
            "webm": self.webm_tasks,
        }

        for tasks in self.indexed_file_tasks.values():
            tasks.define_tasks()

        self.frame_count = 0
        for pose in self.key_frames:
            self.frame_count += pose[0]
        if not self.include_last_frame:
            self.frame_count -= 1
        assert self.frame_count >= 1

        self.rest_image = None

    @property
    def indexed_file_tasks_command_names(self) -> Iterable[str]:
        return self.indexed_file_tasks.keys()

    def get_indexed_file_tasks(self, command_name) -> IndexedFileTasks:
        return self.indexed_file_tasks[command_name]

    def get_pose(self, frame_index):
        key_frame_index = 0
        while frame_index >= self.key_frames[key_frame_index][0]:
            frame_index -= self.key_frames[key_frame_index][0]
            key_frame_index += 1
        if self.key_frames[key_frame_index][0] == 1:
            return self.key_frames[key_frame_index][1].to(self.device).unsqueeze(dim=0)
        else:
            alpha = (frame_index + 1) * 1.0 / self.key_frames[key_frame_index][0]
            last_pose = self.key_frames[key_frame_index - 1][1]
            current_pose = self.key_frames[key_frame_index][1]
            return ((1 - alpha) * last_pose + alpha * current_pose).to(self.device).unsqueeze(dim=0)

    def get_rest_image(self):
        if self.rest_image is None:
            self.rest_image = extract_pytorch_image_from_filelike(self.rest_image_file_name) \
                .to(self.device).unsqueeze(dim=0)
        return self.rest_image

    def frame_file_name(self, frame_index):
        return self.prefix + ("/frame_%08d.png" % frame_index)

    def save_image(self, image, file_name):
        numpy_image = rgba_to_numpy_image(image.detach().squeeze().cpu())
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        print("Saving %s ..." % file_name)
        pil_image.save(file_name)

    def render(self):
        for frame_index in range(self.frame_count):
            pose = self.get_pose(frame_index)
            posed_image = self.poser.pose(self.get_rest_image(), pose)
            self.save_image(posed_image, self.frame_file_name(frame_index))
        with open(self.render_done_tasks.file_name, "wt") as fout:
            fout.write("DONE!!!\n")

    def create_avs_file(self):
        os.makedirs(os.path.dirname(self.avs_tasks.file_name), exist_ok=True)
        with open(self.avs_tasks.file_name, "w") as fout:
            n = self.frame_count
            for i in range(n):
                frame_path = os.path.relpath(self.frame_file_name(i), self.prefix)
                frame_clip = "ImageSource(\"%s\", start = 0, end = 0, fps = 30)" % frame_path
                fout.write(frame_clip)
                if i < n - 1:
                    fout.write(" + \\\n")

    def create_avi_file(self):
        os.makedirs(os.path.dirname(self.avi_tasks.file_name), exist_ok=True)
        convert_avs_to_avi(self.avs_tasks.file_name, self.avi_tasks.file_name)

    def create_mp4_file(self):
        os.makedirs(os.path.dirname(self.mp4_tasks.file_name), exist_ok=True)
        os.system("ffmpeg -i %s -c:v libx264 -preset slow -crf 22 -c:a libfaac -b:a 128k %s" % \
                  (self.avi_tasks.file_name, self.mp4_tasks.file_name))

    def create_webm_file(self):
        os.makedirs(os.path.dirname(self.webm_tasks.file_name), exist_ok=True)
        os.system("ffmpeg -y -i %s -vcodec libvpx -qmin 0 -qmax 50 -crf 10 -b:v 1M -acodec libvorbis %s" % \
                  (self.avi_tasks.file_name, self.webm_tasks.file_name))


class RenderDoneTasks(NoIndexFileTasks):
    def __init__(self, poser_video_tasks: PoserVideoTasksVer2):
        super().__init__(
            poser_video_tasks.workspace,
            poser_video_tasks.prefix,
            "render_done",
            False)
        self.poser_video_tasks = poser_video_tasks

    @property
    def file_name(self):
        return self.prefix + "/render_done.txt"

    def create_file_task(self):
        if self.poser_video_tasks.render_dependencies is not None:
            dependencies = self.poser_video_tasks.render_dependencies
        else:
            dependencies = []
        self.workspace.create_file_task(
            self.file_name,
            dependencies,
            self.poser_video_tasks.render)


class AvsTasks(NoIndexFileTasks):
    def __init__(self, poser_video_tasks: PoserVideoTasksVer2):
        super().__init__(
            poser_video_tasks.workspace,
            poser_video_tasks.prefix,
            "avs",
            False)
        self.poser_video_tasks = poser_video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.avs"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [],
            self.poser_video_tasks.create_avs_file)


class AviTasks(NoIndexFileTasks):
    def __init__(self, poser_video_tasks: PoserVideoTasksVer2):
        super().__init__(
            poser_video_tasks.workspace,
            poser_video_tasks.prefix,
            "avi",
            False)
        self.poser_video_tasks = poser_video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.avi"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [
                self.poser_video_tasks.render_done_tasks.file_name,
                self.poser_video_tasks.avs_tasks.file_name,
            ],
            self.poser_video_tasks.create_avi_file)


class Mp4Tasks(NoIndexFileTasks):
    def __init__(self, poser_video_tasks: PoserVideoTasksVer2):
        super().__init__(
            poser_video_tasks.workspace,
            poser_video_tasks.prefix,
            "mp4",
            False)
        self.poser_video_tasks = poser_video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.mp4"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [
                self.poser_video_tasks.avi_tasks.file_name,
            ],
            self.poser_video_tasks.create_mp4_file)


class WebmTasks(NoIndexFileTasks):
    def __init__(self, poser_video_tasks: PoserVideoTasksVer2):
        super().__init__(
            poser_video_tasks.workspace,
            poser_video_tasks.prefix,
            "webm",
            False)
        self.poser_video_tasks = poser_video_tasks

    @property
    def file_name(self):
        return self.prefix + "/video.webm"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [
                self.poser_video_tasks.avi_tasks.file_name,
            ],
            self.poser_video_tasks.create_webm_file)
