import os
from typing import Callable, List

import PIL.Image
import numpy
import torch

from hana.rindou.poser.poser.poser import Poser
from hana.rindou.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image
from hana.rindou.video.image_sequence_tasks import ImageSequenceTasks
from pytasuku import Workspace


class PoserImageSequenceTasks(ImageSequenceTasks):
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 rest_image_file_name: str,
                 poser: Poser,
                 frame_count: int,
                 pose_func: Callable[[int], torch.Tensor],
                 render_dependencies: List[str] = None,
                 device: torch.device = torch.device("cpu")):
        super().__init__(workspace, prefix, frame_count, render_dependencies)

        self.device = device
        self.poser = poser
        self.pose_func = pose_func
        self.rest_image_file_name = rest_image_file_name
        self.rest_image = None

    def get_pose(self, frame_index):
        return self.pose_func(frame_index).to(self.device)

    def get_rest_image(self):
        if self.rest_image is None:
            self.rest_image = extract_pytorch_image_from_filelike(self.rest_image_file_name) \
                .to(self.device).unsqueeze(dim=0)
        return self.rest_image

    def save_image(self, image, file_name):
        numpy_image = rgba_to_numpy_image(image.detach().squeeze().cpu())
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pil_image.save(file_name)

    def render_frame(self, frame_index):
        pose = self.get_pose(frame_index)
        posed_image = self.poser.pose(self.get_rest_image(), pose)
        self.save_image(posed_image, self.frame_file_name(frame_index))