import os

import torch

from torch.utils.data import Dataset

from hana.rindou.poser.dataset.three_step_data import load_three_step_data_tsv
from hana.rindou.util import extract_pytorch_image_from_filelike


class MorphRegressionFromThreeStepDataDataset(Dataset):
    def __init__(self, data_tsv_file_name: str,
                 image_size: int = 256,
                 pose_size: int = 6,
                 bone_parameters_count: int = 3,
                 verbose: bool = False,
                 paths_are_relative: bool = False):
        self.data_tsv_file_name = data_tsv_file_name
        self.examples = None
        self.pose_size = pose_size
        self.bone_parameters_count = bone_parameters_count
        self.verbose = verbose
        self.image_size = image_size
        self.paths_are_relative = paths_are_relative

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
        if rest_image.shape[1] != self.image_size or rest_image.shape[2] != self.image_size:
            raise RuntimeError(example[0])
        morphed_image = self.load_image(example[2])
        if morphed_image.shape[1] != self.image_size or morphed_image.shape[2] != self.image_size:
            raise RuntimeError(example[2])
        _pose = example[1][self.bone_parameters_count:]
        pose = torch.Tensor(_pose)
        return [rest_image, morphed_image, pose]

    def load_image(self, file_name):
        if self.paths_are_relative:
            file_name = os.path.dirname(self.data_tsv_file_name) + "/" + file_name
        if self.verbose:
            print("Loading %s ..." % file_name)
        with open(file_name, "rb") as file:
            return extract_pytorch_image_from_filelike(file)


if __name__ == "__main__":
    dataset = MorphRegressionFromThreeStepDataDataset(
        "data/rindou/_20190906/three_step/train/data.tsv",
        pose_size=6,
        bone_parameters_count=3,
        verbose=True)
    example = dataset[0]
    print(example[0].shape)
    print(example[1].shape)
    print(example[2].shape)