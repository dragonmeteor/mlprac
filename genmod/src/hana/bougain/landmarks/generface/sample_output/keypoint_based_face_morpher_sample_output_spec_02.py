import os
from typing import List

import PIL.Image
import math
import numpy
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torchvision.transforms.functional import adjust_hue

from hana.bougain.landmarks.dataset.expression_pair_data_from_three_steps import \
    ExpressionPairDatasetFromThreeStepsWithAugmentation
from hana.bougain.landmarks.generface.keypoint_based_face_morpher_01 import KeypointBasedFaceMorpher01
from hana.rindou.poser.v1.poser_gan_tasks_ver2 import PoserGanSampleOutputSpecVer2
from hana.rindou.util import rgb_to_numpy_image, rgba_to_numpy_image, torch_load


class KeypointBasedFaceMorpherSampledOutputSpec02(PoserGanSampleOutputSpecVer2):
    def __init__(self, example_per_sample_output=5000):
        super().__init__()
        self._example_per_sample_output = example_per_sample_output

    @property
    def example_per_row(self) -> int:
        return 1

    @property
    def example_per_sample_output(self) -> int:
        return self._example_per_sample_output

    @property
    def sample_output_index_seed(self) -> int:
        return 147258399

    @property
    def image_size(self) -> int:
        return 256

    @property
    def count(self) -> int:
        return 8

    def resize_and_detach_image(self, image):
        return interpolate(image, size=self.image_size).cpu().detach()

    def get_data_bundle(self, G, source_image, target_image):
        output = G(source_image, target_image)
        return {
            "source_image": self.resize_and_detach_image(source_image),
            "target_image": self.resize_and_detach_image(target_image),
            "reconstructed_target_image": self.resize_and_detach_image(output[0]),
            "alpha": self.resize_and_detach_image(output[1]),
            "color_change": self.resize_and_detach_image(output[2]),
            "keypoints": output[3].cpu().detach(),
            "heatmap": self.resize_and_detach_image(output[4])
        }

    def get_heatmap_sum_image(self, heatmap):
        heatmap = heatmap.sum(dim=0, keepdim=True)
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        return heatmap.repeat(3, 1, 1)

    def draw_keypoints_on_image(self, keypoints, base_image, color=(0.0, 1.0, 0.0)):
        c, h, w = base_image.shape
        keypoint_image = base_image.clone()
        num_keypoints = keypoints.shape[0]
        for j in range(num_keypoints):
            xy = (keypoints[j] + 1.0) / 2.0
            y = int(math.floor(xy[1].item() * h))
            x = int(math.floor(xy[0].item() * w))
            for dx in range(2):
                for dy in range(2):
                    self.set_pixel(keypoint_image, x + dx, y + dy, color)
        return keypoint_image

    def set_pixel(self, image, x, y, color):
        c, h, w = image.shape
        if x >= 0 and x < w and y >= 0 and y < h:
            image[0, y, x] = color[0]
            image[1, y, x] = color[1]
            image[2, y, x] = color[2]
            if c == 4:
                image[3, y, x] = 1.0

    def save_sample_image(self, G: Module, sample_output_batch: List[Tensor], file_name: str):
        G.train(False)
        n = sample_output_batch[0].shape[0]
        original_data_bundle = self.get_data_bundle(G, sample_output_batch[0], sample_output_batch[1])
        augmented_data_bundle = self.get_data_bundle(G, sample_output_batch[2], sample_output_batch[3])
        inverse_matrices = sample_output_batch[5].cpu().detach()

        num_rows = n
        # (1) source_image,
        # (2) target_image,
        # (3) reconstructed_target_image
        # (4) alpha,
        # (5) color_change,
        # (6) keypoints
        num_cols = 8

        image_size = self.image_size
        output_image = numpy.zeros([num_rows * image_size, num_cols * image_size, 3])
        has_alpha = original_data_bundle["source_image"].shape[1] != 3

        for i in range(n):
            self.set_output_subimage(output_image, i, 0,
                                     original_data_bundle["source_image"][i], has_alpha=has_alpha)
            self.set_output_subimage(output_image, i, 1,
                                     original_data_bundle["target_image"][i], has_alpha=has_alpha)
            self.set_output_subimage(output_image, i, 2,
                                     original_data_bundle["reconstructed_target_image"][i], has_alpha=has_alpha)
            self.set_output_subimage(output_image, i, 3,
                                     original_data_bundle["alpha"][i].repeat(3, 1, 1), has_alpha=False)
            self.set_output_subimage(output_image, i, 4,
                                     original_data_bundle["color_change"][i], has_alpha=True)
            self.set_output_subimage(output_image, i, 5,
                                     self.get_heatmap_sum_image(original_data_bundle["heatmap"][i]), has_alpha=False)
            self.set_output_subimage(output_image, i, 6,
                                     self.draw_keypoints_on_image(
                                         original_data_bundle["keypoints"][i],
                                         original_data_bundle["target_image"][i] * 0.5,
                                         color=(0.0, 1.0, 0.0)),
                                     has_alpha=has_alpha)

            original_keypoints = original_data_bundle["keypoints"][i]
            k = original_keypoints.shape[0]
            ones = torch.ones(k, 1)
            original_keypoints = torch.cat([original_keypoints, ones], dim=1).view(k, 3, 1)
            inverse_matrix = inverse_matrices[i].view(1,2,3)
            xformed_keypoints = torch.matmul(inverse_matrix, original_keypoints).view(k, 2)

            keypoint_image = self.draw_keypoints_on_image(
                xformed_keypoints,
                augmented_data_bundle["target_image"][i] * 0.5,
                color=(1.0, 0.0, 1.0))
            keypoint_image = self.draw_keypoints_on_image(
                augmented_data_bundle["keypoints"][i],
                keypoint_image,
                color=(0.0, 1.0, 0.0))

            self.set_output_subimage(output_image, i, 7, keypoint_image, has_alpha=has_alpha)

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(output_image * 255.0)), mode='RGB')
        pil_image.save(file_name)
        print("Saved %s" % file_name)

    def convert_to_numpy_image(self, image, has_alpha: bool = True):
        if not has_alpha:
            image = rgb_to_numpy_image(image, min_pixel_value=0.0, max_pixel_value=1.0)
        else:
            image = rgba_to_numpy_image(image, min_pixel_value=0.0, max_pixel_value=1.0)
            image = image[:, :, 0:3] * image[:, :, 3:4]
        return image

    def set_output_subimage(self, output, row, col, image, has_alpha: bool = True):
        image = self.convert_to_numpy_image(image, has_alpha)
        output[
        row * self.image_size:(row + 1) * self.image_size,
        col * self.image_size:(col + 1) * self.image_size,
        :] \
            = image


if __name__ == "__main__":
    cuda = torch.device('cuda')
    face_morpher = KeypointBasedFaceMorpher01(
        in_channels=4,
        num_keypoints=14,
        keypoint_detector_in_channels=16,
        decomposer_in_channels=16,
        keypoint_detector_spec='zhang').to(cuda)
    face_morpher.load_state_dict(torch_load("data/bougain/_20200715/keypoint_based_face_morpher_03/generator_001.pt"))

    validation_set = ExpressionPairDatasetFromThreeStepsWithAugmentation(
        data_tsv_file_name="data/rindou/_20190906/three_step_varying_ambience/validation/data.tsv",
        has_alpha=True)
    data_loader = DataLoader(validation_set, batch_size=8, shuffle=True, num_workers=6, drop_last=True)
    batch = next(data_loader.__iter__())
    batch = [x.to(cuda) for x in batch]

    sampled_output_spec = KeypointBasedFaceMorpherSampledOutputSpec02()
    sampled_output_spec.save_sample_image(
        face_morpher,
        batch,
        "data/bougain/_20200715/temp.png")
