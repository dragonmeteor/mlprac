import os

import PIL.Image
from functools import reduce

import numpy
import torch

from torch import Tensor
from typing import List, Tuple

from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec
from hana.rindou.util import extract_pytorch_image_from_filelike, torch_load, rgba_to_numpy_image, convert_avs_to_avi
from pytasuku import Workspace


class PoserGanVideoTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 rest_image_file_name: str,
                 start_pose: Tensor,
                 pose_sequence: List,
                 gan_spec: PoserGanSpec,
                 generator_file_name: str,
                 device: torch.device = torch.device('cpu')):
        self.workspace = workspace
        self.prefix = prefix
        self.rest_image_file_name = rest_image_file_name
        self.start_pose = start_pose
        self.pose_sequence = pose_sequence
        self.gan_spec = gan_spec
        self.generator_file_name = generator_file_name
        self.device = device

        self.rest_image = None
        self.G = None

        self.define_tasks()

    def load_rest_image(self):
        if self.rest_image is None:
            with open(self.rest_image_file_name, "rb") as file:
                self.rest_image = extract_pytorch_image_from_filelike(file).to(self.device).unsqueeze(0)

    def load_generator(self):
        if self.G is None:
            self.G = self.gan_spec.generator().to(self.device)
            self.G.load_state_dict(torch_load(self.generator_file_name))

    def get_frame_count(self):
        return reduce(lambda x, y: x + y, map(lambda x: x[0], self.pose_sequence), 0) + 1

    def frame_file_name(self, index):
        return self.prefix + ("/frame_%08d.png" % index)

    def frame_done_file_name(self):
        return self.prefix + "/frame_done.txt"

    def define_tasks(self):
        self.workspace.create_file_task(
            self.frame_done_file_name(),
            [],
            lambda: self.render_frames())
        self.workspace.create_file_task(
            self.avs_file_name(),
            [],
            lambda: self.create_avs())
        self.workspace.create_file_task(
            self.avi_file_name(),
            [
                self.avs_file_name(),
                self.frame_done_file_name()
            ],
            lambda: self.create_avi())
        self.workspace.create_file_task(
            self.mp4_file_name(),
            [self.avi_file_name()],
            lambda: self.create_mp4())

    def render_frames(self):
        self.load_rest_image()
        self.load_generator()
        self.G.train(False)
        pose0 = self.start_pose.to(self.device).unsqueeze(0)
        self.render_frame(pose0, 0)
        frame_index = 0
        for i in range(len(self.pose_sequence)):
            frame_count = self.pose_sequence[i][0]
            pose1 = self.pose_sequence[i][1].to(self.device).unsqueeze(0)
            for j in range(frame_count):
                alpha = (j + 1) * 1.0 / frame_count
                pose = (1.0 - alpha) * pose0 + alpha * pose1
                frame_index += 1
                self.render_frame(pose, frame_index)
            pose0 = pose1

        os.makedirs(os.path.dirname(self.frame_done_file_name()), exist_ok=True)
        with open(self.frame_done_file_name(), "wt") as f:
            f.write("DONE!!!")

    def render_frame(self, pose, frame_index):
        output_image = self.G(self.rest_image, pose)[0].detach().cpu().squeeze()
        numpy_image = rgba_to_numpy_image(output_image)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
        os.makedirs(os.path.dirname(self.frame_file_name(frame_index)), exist_ok=True)
        print("Saving %s ..." % self.frame_file_name(frame_index))
        pil_image.save(self.frame_file_name(frame_index))

    def avs_file_name(self):
        return self.prefix + "/video.avs"

    def create_avs(self):
        os.makedirs(os.path.dirname(self.avs_file_name()), exist_ok=True)
        with open(self.avs_file_name(), "w") as fout:
            n = self.get_frame_count()
            for i in range(n):
                rest_path = os.path.relpath(self.rest_image_file_name, self.prefix)
                rest_clip = "Subtitle(ImageSource(\"%s\", start = 0, end = 0, fps = 30), \"Original\", align=8)" % rest_path
                frame_path = os.path.relpath(self.frame_file_name(i), self.prefix)
                frame_clip = "Subtitle(ImageSource(\"%s\", start = 0, end = 0, fps = 30), \"Network generated animation\", align=8)" % frame_path
                stack_clip = "StackHorizontal(%s, %s)" % (rest_clip, frame_clip)
                fout.write(stack_clip)
                if i < n - 1:
                    fout.write(" + \\\n")
            # rest_image_path = os.path.relpath(self.rest_image_file_name, self.prefix)
            # rest_image = "ImageSource(%s, start=0, end=0, fps=30)" % rest_image_path

    def avi_file_name(self):
        return self.prefix + "/video.avi"

    def create_avi(self):
        os.makedirs(os.path.dirname(self.avi_file_name()), exist_ok=True)
        convert_avs_to_avi(self.avs_file_name(), self.avi_file_name())

    def mp4_file_name(self):
        return self.prefix + "/video.mp4"

    def create_mp4(self):
        os.makedirs(os.path.dirname(self.mp4_file_name()), exist_ok=True)
        os.system("ffmpeg -i %s -c:v libx264 -preset slow -crf 22 -c:a libfaac -b:a 128k %s" %\
                  (self.avi_file_name(), self.mp4_file_name()))
