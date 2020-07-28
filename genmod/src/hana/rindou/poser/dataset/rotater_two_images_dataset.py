import csv
import os

import torch
from torch.utils.data import Dataset

from hana.rindou.util import extract_pytorch_image_from_filelike


def load_rotater_two_images_data_tsv(data_tsv_file_name: str, pose_size=6):
    print("Loading", data_tsv_file_name, "...")
    examples = []
    data_tsv_file = open(data_tsv_file_name)
    tsvreader = csv.reader(data_tsv_file, delimiter='\t')

    for line in tsvreader:
        image_0_file_name = line[0]
        image_1_file_name = line[1]
        pose = [float(x) for x in line[2:pose_size + 2]]
        source_image_file_name = line[pose_size + 2]
        target_image_file_name = line[pose_size + 3]

        example = [
            image_0_file_name,
            image_1_file_name,
            pose,
            source_image_file_name,
            target_image_file_name
        ]

        examples.append(example)

    data_tsv_file.close()
    print("Loading", data_tsv_file_name, "done!!!")
    return examples


class RotaterTwoImagesDataset(Dataset):
    def __init__(self,
                 data_tsv_file_name: str,
                 image_size: int = 256,
                 pose_size: int = 6,
                 bone_parameters_count: int = 3,
                 verbose: bool = False,
                 load_source_image: bool = False,
                 paths_are_relative: bool = False):
        self.data_tsv_file_name = data_tsv_file_name
        self.examples = None
        self.pose_size = pose_size
        self.bone_parameters_count = bone_parameters_count
        self.verbose = verbose
        self.image_size = image_size
        self.load_source_image = load_source_image
        self.paths_are_relative = paths_are_relative

    def get_examples(self):
        if self.examples is None:
            examples = load_rotater_two_images_data_tsv(self.data_tsv_file_name, self.pose_size)
            self.examples = examples
        return self.examples

    def __len__(self):
        return len(self.get_examples())

    def __getitem__(self, index):
        examples = self.get_examples()
        example = examples[index]
        image_0 = self.load_image(example[0])
        image_1 = self.load_image(example[1])
        target_image = self.load_image(example[4])
        _pose = example[2][:self.bone_parameters_count]
        pose = torch.Tensor(_pose)
        if self.load_source_image:
            source_image = self.load_image(example[3])
            return [image_0, image_1, pose, source_image, target_image]
        else:
            return [image_0, image_1, pose, target_image]

    def load_image(self, file_name):
        if self.paths_are_relative:
            file_name = os.path.dirname(self.data_tsv_file_name) + "/" + file_name
        if self.verbose:
            print("Loading %s ..." % file_name)
        with open(file_name, "rb") as file:
            return extract_pytorch_image_from_filelike(file)
