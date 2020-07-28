from typing import List

import torch
from torch import Tensor

from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec
from hana.rindou.util import torch_load
from hana.rindou.poser.poser.poser import Poser, PoseParameter


class MorphThenPoseGanPoser(Poser):
    def __init__(self,
                 morph_gan_spec: PoserGanSpec,
                 morph_gan_generator_file_name: str,
                 morph_gan_parameters: List[PoseParameter],
                 pose_gan_spec: PoserGanSpec,
                 pose_gan_generator_file_name: str,
                 pose_gan_parameters: List[PoseParameter],
                 device: torch.device):
        self.morph_gan_spec = morph_gan_spec
        self.morph_gan_generator_file_name = morph_gan_generator_file_name
        self.morph_gan_parameters = morph_gan_parameters
        self.pose_gan_spec = pose_gan_spec
        self.pose_gan_generator_file_name = pose_gan_generator_file_name
        self.pose_gan_parameters = pose_gan_parameters
        self.device = device

        self.morph_generator = None
        self.pose_generator = None

    def pose_parameters(self) -> List[PoseParameter]:
        return self.pose_gan_parameters + self.morph_gan_parameters

    def get_morph_generator(self):
        if self.morph_generator is None:
            G = self.morph_gan_spec.generator().to(self.device)
            G.load_state_dict(torch_load(self.morph_gan_generator_file_name))
            self.morph_generator = G
            G.train(False)
        return self.morph_generator

    def get_pose_generator(self):
        if self.pose_generator is None:
            G = self.pose_gan_spec.generator().to(self.device)
            G.load_state_dict(torch_load(self.pose_gan_generator_file_name))
            self.pose_generator = G
            G.train(False)
        return self.pose_generator

    def pose(self, source_image: Tensor, pose: Tensor):
        morph_param_count = len(self.morph_gan_parameters)
        pose_param_count = len(self.pose_gan_parameters)

        morph_params = pose[:, morph_param_count:morph_param_count + pose_param_count]
        pose_params = pose[:, 0:morph_param_count]

        morph_generator = self.get_morph_generator()
        morphed_image = morph_generator(source_image, morph_params)[0]


        pose_generator = self.get_pose_generator()
        posed_image = pose_generator(morphed_image, pose_params)[0]

        return posed_image


class Rindou00MorphThenPoseGanPoser(MorphThenPoseGanPoser):
    def __init__(self,
                 morph_gan_spec: PoserGanSpec,
                 morph_gan_generator_file_name: str,
                 pose_gan_spec: PoserGanSpec,
                 pose_gan_generator_file_name: str,
                 device: torch.device):
        super().__init__(
            morph_gan_spec,
            morph_gan_generator_file_name,
            [
                PoseParameter("left_eye", "Left Eye", 0.0, 1.0, 0.0),
                PoseParameter("right_eye", "Right Eye", 0.0, 1.0, 0.0),
                PoseParameter("mouth", "Mouth", 0.0, 1.0, 1.0)
            ],
            pose_gan_spec,
            pose_gan_generator_file_name,
            [
                PoseParameter("head_x", "Head X", -1.0, 1.0, 0.0),
                PoseParameter("head_y", "Head Y", -1.0, 1.0, 0.0),
                PoseParameter("neck_z", "Neck Z", -1.0, 1.0, 0.0),
            ],
            device)

    def image_size(self):
        return self.morph_gan_spec.image_size()