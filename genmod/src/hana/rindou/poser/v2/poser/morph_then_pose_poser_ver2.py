from abc import ABC
from typing import List

import torch
from torch import Tensor

from hana.rindou.poser.poser.poser import Poser, PoseParameter
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec
from hana.rindou.util import torch_load


class MorphThenPosePoserVer2(Poser):
    def __init__(self,
                 morph_module_spec: PoserGanModuleSpec,
                 morph_module_file_name: str,
                 morph_module_parameters: List[PoseParameter],
                 pose_module_spec: PoserGanModuleSpec,
                 pose_module_file_name: str,
                 pose_module_parameters: List[PoseParameter],
                 image_size: int,
                 device: torch.device):
        self.morph_module_spec = morph_module_spec
        self.morph_module_file_name = morph_module_file_name
        self.morph_module_parameters = morph_module_parameters
        self.pose_module_spec = pose_module_spec
        self.pose_module_file_name = pose_module_file_name
        self.pose_module_parameters = pose_module_parameters
        self.device = device
        self._image_size = image_size

        self.morph_module = None
        self.pose_module = None

    def image_size(self):
        return self._image_size

    def pose_parameters(self) -> List[PoseParameter]:
        return self.pose_module_parameters + self.morph_module_parameters

    def get_morph_module(self):
        if self.morph_module is None:
            G = self.morph_module_spec.get_module().to(self.device)
            G.load_state_dict(torch_load(self.morph_module_file_name))
            self.morph_module = G
            G.train(False)
        return self.morph_module

    def get_pose_module(self):
        if self.pose_module is None:
            G = self.pose_module_spec.get_module().to(self.device)
            G.load_state_dict(torch_load(self.pose_module_file_name))
            self.pose_module = G
            G.train(False)
        return self.pose_module

    def pose(self, source_image: Tensor, pose: Tensor):
        morph_param_count = len(self.morph_module_parameters)
        pose_param_count = len(self.pose_module_parameters)

        morph_params = pose[:, morph_param_count:morph_param_count + pose_param_count]
        pose_params = pose[:, 0:morph_param_count]

        morph_generator = self.get_morph_module()
        morphed_image = morph_generator(source_image, morph_params)[0]

        pose_generator = self.get_pose_module()
        posed_image = pose_generator(morphed_image, pose_params)[0]

        return posed_image

class Rindou00MorphThenPosePoserVer2(MorphThenPosePoserVer2):
    def __init__(self,
                 morph_module_spec: PoserGanModuleSpec,
                 morph_module_file_name: str,
                 pose_module_spec: PoserGanModuleSpec,
                 pose_module_file_name: str,
                 device: torch.device):
        super().__init__(
            morph_module_spec,
            morph_module_file_name,
            [
                PoseParameter("left_eye", "Left Eye", 0.0, 1.0, 0.0),
                PoseParameter("right_eye", "Right Eye", 0.0, 1.0, 0.0),
                PoseParameter("mouth", "Mouth", 0.0, 1.0, 1.0)
            ],
            pose_module_spec,
            pose_module_file_name,
            [
                PoseParameter("head_x", "Head X", -1.0, 1.0, 0.0),
                PoseParameter("head_y", "Head Y", -1.0, 1.0, 0.0),
                PoseParameter("neck_z", "Neck Z", -1.0, 1.0, 0.0),
            ],
            256,
            device)