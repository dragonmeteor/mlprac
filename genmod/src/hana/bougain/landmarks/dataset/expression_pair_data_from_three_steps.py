import csv
from typing import Tuple

import PIL.Image
import numpy
import torch
from torch.nn.functional import affine_grid, grid_sample
from torch.utils.data import Dataset
from torchvision.transforms.functional import adjust_hue, adjust_contrast, adjust_saturation, adjust_brightness

from hana.rindou.poser.dataset.three_step_data import load_three_step_data_tsv
from hana.rindou.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image, \
    extract_pytorch_image_from_PIL_image


class ExpressionPairDatasetFromThreeSteps(Dataset):
    def __init__(self, data_tsv_file_name, has_alpha=False, verbose=False):
        self.has_alpha = has_alpha
        self.data_tsv_file_name = data_tsv_file_name
        self.verbose = verbose
        self.examples = None

    def get_examples(self):
        if self.examples is None:
            examples = load_three_step_data_tsv(self.data_tsv_file_name)
            self.examples = examples
        return self.examples

    def __len__(self):
        return len(self.get_examples())

    def __getitem__(self, index):
        examples = self.get_examples()
        example = examples[index]
        rest_image = self.load_image(example[0])
        morphed_image = self.load_image(example[2])
        return [rest_image, morphed_image]

    def load_image(self, file_name):
        with open(file_name, "rb") as file:
            return extract_pytorch_image_from_filelike(file, scale=1.0, offset=0.0)


class ExpressionPairDatasetFromThreeStepsWithAugmentation(Dataset):
    def __init__(self,
                 data_tsv_file_name: str,
                 has_alpha=False,
                 diag_range: Tuple[float,float] = (0.8, 1.2),
                 off_diag_range: Tuple[float,float] = (-0.4, 0.4),
                 translation_range: Tuple[float,float] = (-0.2, 0.2),
                 brightness_range: Tuple[float,float] = (0.9, 1.1),
                 saturation_range: Tuple[float,float] = (0.9, 1.1),
                 hue_range: Tuple[float,float] = (0.0, 0.0),
                 verbose=False):
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.brightness_range = brightness_range
        self.translation_range = translation_range
        self.off_diag_range = off_diag_range
        self.diag_range = diag_range
        self.has_alpha = has_alpha
        self.data_tsv_file_name = data_tsv_file_name
        self.verbose = verbose
        self.examples = None

    def get_examples(self):
        if self.examples is None:
            examples = load_three_step_data_tsv(self.data_tsv_file_name)
            self.examples = examples
        return self.examples

    def __len__(self):
        return len(self.get_examples())

    def __getitem__(self, index):
        examples = self.get_examples()
        example = examples[index]
        image_params = torch.rand(3)
        rest_image = self.load_image(example[0], image_params)
        morphed_image = self.load_image(example[2], image_params)
        matrix = self.get_matrix()
        augmented_rest_image = self.resample_image(matrix, rest_image)
        augmented_morphed_image = self.resample_image(matrix, morphed_image)
        inverse = torch.inverse(torch.cat([matrix, torch.tensor([[0.0, 0.0, 1.0]])], dim=0))[0:2, :]
        return [rest_image, morphed_image, augmented_rest_image, augmented_morphed_image, matrix, inverse]

    def get_matrix(self):
        xi = torch.rand((2, 3))
        offset = torch.tensor([
            [self.diag_range[0], self.off_diag_range[0], self.translation_range[0]],
            [self.off_diag_range[0], self.diag_range[0], self.translation_range[0]]
        ])

        scale = torch.tensor([
            [
                self.diag_range[1] - self.diag_range[0],
                self.off_diag_range[1] - self.off_diag_range[0],
                self.translation_range[1] - self.translation_range[0]
            ],
            [
                self.off_diag_range[1] - self.off_diag_range[0],
                self.diag_range[1] - self.diag_range[0],
                self.translation_range[1] - self.translation_range[0]
            ]
        ])
        return offset + scale * xi

    def resample_image(self, matrix, image):
        c, h, w = image.shape
        grid = affine_grid(matrix.view(1, 2, 3), [1, c, h, w], align_corners=False)
        modded_image = grid_sample(
            image.view(1, c, h, w),
            grid, mode='bilinear',
            padding_mode='border',
            align_corners=False)
        return modded_image.view(c, h, w)

    def load_image(self, file_name, image_params):
        with open(file_name, "rb") as file:
            image = PIL.Image.open(file)
            if self.has_alpha:
                image = numpy.asarray(image)
                image_alpha = image[:,:,3:4]
                image = image[:,:,0:3]
                image = PIL.Image.fromarray(image)
            image = adjust_brightness(
                image,
                self.brightness_range[0] + (self.brightness_range[1] - self.brightness_range[0]) * image_params[
                    0].item())
            image = adjust_saturation(
                image,
                self.saturation_range[0] + (self.saturation_range[1] - self.saturation_range[0]) * image_params[
                    1].item())
            image = adjust_hue(
                image,
                self.hue_range[0] + (self.hue_range[1] - self.hue_range[0]) * image_params[2].item())
            if self.has_alpha:
                image = numpy.asarray(image)
                image = numpy.concatenate((image, image_alpha), axis=2)
                image = PIL.Image.fromarray(image)
            return extract_pytorch_image_from_PIL_image(image, scale=1.0, offset=0.0)


def load_data_tsv_file(data_tsv_file_name):
    examples = []
    data_tsv_file = open(data_tsv_file_name)
    tsvreader = csv.reader(data_tsv_file, delimiter='\t')
    for line in tsvreader:
        example = []
        example.append(line[0])
        example.append(line[1])
        examples.append(example)
    data_tsv_file.close()
    return examples


def save_image(image, file_name):
    import PIL.Image
    import numpy
    image = rgba_to_numpy_image(image, min_pixel_value=0.0, max_pixel_value=1.0)
    pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(image * 255.0)), mode='RGBA')
    pil_image.save(file_name)
    print("Saved %s" % file_name)


if __name__ == "__main__":
    dataset = ExpressionPairDatasetFromThreeStepsWithAugmentation(
        "data/rindou/_20190906/three_step_varying_ambience/validation/data.tsv",
        has_alpha=True)
    example = dataset[2434]
    print(example[0].shape)
    print(example[1].shape)
    print(example[2].shape)
    print(example[3].shape)
    print(example[4].shape)

    save_image(example[2], "data/bougain/_20200715/rest_xformed.png")
    save_image(example[3], "data/bougain/_20200715/morphed_xformed.png")
