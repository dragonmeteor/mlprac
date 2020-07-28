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

from hana.bougain.landmarks.dataset.expression_pair_data_from_three_steps import ExpressionPairDatasetFromThreeSteps
from hana.bougain.landmarks.generface.keypoint_based_face_morpher_01 import KeypointBasedFaceMorpher01
from hana.rindou.poser.v1.poser_gan_tasks_ver2 import PoserGanSampleOutputSpecVer2
from hana.rindou.util import rgb_to_numpy_image, rgba_to_numpy_image, torch_load


class KeypointBasedFaceMorpherSampledOutputSpec01(PoserGanSampleOutputSpecVer2):
    @property
    def example_per_row(self) -> int:
        return 1

    @property
    def example_per_sample_output(self) -> int:
        return 5000

    @property
    def sample_output_index_seed(self) -> int:
        return 147258399

    @property
    def image_size(self) -> int:
        return 256

    @property
    def count(self) -> int:
        return 8

    def save_sample_image(self, G: Module, sample_output_batch: List[Tensor], file_name: str):
        G.train(False)
        source_images = sample_output_batch[0]
        target_images = sample_output_batch[1]
        output = G(source_images, target_images)

        source_images = interpolate(source_images, size=self.image_size).cpu()
        target_images = interpolate(target_images, size=self.image_size).cpu()
        output = [interpolate(x, size=self.image_size).cpu().detach() for x in output[:3]] \
                 + [x.cpu().detach() for x in output[3:4]] \
                 + [interpolate(x, size=self.image_size).cpu().detach() for x in output[4:]]
        n = source_images.shape[0]
        num_rows = n
        # (1) source_image,
        # (2) target_image,
        # (3) reconstructed_target_image
        # (4) alpha,
        # (5) color_change,
        # (6) keypoints
        num_cols = 7

        image_size = self.image_size
        output_image = numpy.zeros([num_rows * image_size, num_cols * image_size, 3])
        has_alpha = source_images[0].shape[1] != 3

        for i in range(n):
            heatmap = output[4][i].sum(dim=0, keepdim=True)
            heatmap_min = heatmap.min()
            heatmap_max = heatmap.max()
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

            self.set_output_subimage(output_image, i, 0, source_images[i], has_alpha=has_alpha)
            self.set_output_subimage(output_image, i, 1, target_images[i], has_alpha=has_alpha)
            self.set_output_subimage(output_image, i, 2, output[0][i], has_alpha=has_alpha)
            self.set_output_subimage(output_image, i, 3, output[1][i].repeat(3, 1, 1), has_alpha=False)
            self.set_output_subimage(output_image, i, 4, output[2][i], has_alpha=True)
            self.set_output_subimage(output_image, i, 6, heatmap.repeat(3, 1, 1), has_alpha=False)

            # keypoint_image = numpy.zeros([self.image_size, self.image_size, 3])
            keypoint_image = self.convert_to_numpy_image(target_images[i] * 0.5)
            keypoints = output[3][i]
            num_keypoints = keypoints.shape[0]
            for j in range(num_keypoints):
                xy = (keypoints[j] + 1.0) / 2.0 * 256
                x = int(math.floor(xy[1].item()))
                y = int(math.floor(xy[0].item()))
                for dx in range(2):
                    for dy in range(2):
                        self.set_pixel_to_green(keypoint_image, x + dx, y + dy)
            output_image[
            i * self.image_size:(i + 1) * self.image_size,
            5 * self.image_size:(5 + 1) * self.image_size,
            :] \
                = keypoint_image

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(output_image * 255.0)), mode='RGB')
        pil_image.save(file_name)
        print("Saved %s" % file_name)

    def set_pixel_to_green(self, image, x, y):
        w, h, _ = image.shape
        if x >= 0 and x < w and y >= 0 and y < h:
            image[x, y, 1] = 1.0

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
        num_keypoints=16,
        in_channels=4,
        decomposer_in_channels=16,
        keypoint_detector_in_channels=16,
        num_blocks=5,
        max_channels=1024,
        keypoint_variance=0.01,
        activation='relu').to(cuda)
    face_morpher.load_state_dict(torch_load("data/bougain/_20200715/keypoint_based_face_morpher_02/generator_001.pt"))

    validation_set = ExpressionPairDatasetFromThreeSteps(
        data_tsv_file_name="data/rindou/_20190906/three_step_varying_ambience/validation/data.tsv",
        has_alpha=True)
    data_loader = DataLoader(validation_set, batch_size=8, shuffle=True, num_workers=6, drop_last=True)
    batch = next(data_loader.__iter__())
    batch = [x.to(cuda) for x in batch]

    sampled_output_spec = KeypointBasedFaceMorpherSampledOutputSpec01()
    sampled_output_spec.save_sample_image(
        face_morpher,
        batch,
        "data/bougain/_20200715/temp.png")
