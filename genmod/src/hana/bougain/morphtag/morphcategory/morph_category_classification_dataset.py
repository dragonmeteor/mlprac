import os

import torch
from torch.utils.data import Dataset

import json

from hana.rindou.util import extract_pytorch_image_from_filelike


class MorphCategoryClassificationDataset(Dataset):
    def __init__(self,
                 data_json_file_name: str,
                 verbose: bool = False):
        self.data_json_file_name = data_json_file_name
        self.verbose = verbose

        self.examples = None
        self.morph_category_enum_names = None
        self.panel_enum_names = None
        self.image_size = None
        self.name_vector_size = None

    def load_data(self):
        with open(self.data_json_file_name, "rt", encoding="utf-8") as json_file:
            data = json.load(json_file)
        self.morph_category_enum_names = data["morphCategoryEnumNames"]
        self.panel_enum_names = data["panelEnumNames"]
        self.image_size = data["imageSize"]
        self.name_vector_size = data["nameVectorSize"]
        self.examples = data["examples"]

    def get_examples(self):
        if self.examples is None:
            self.load_data()
        return self.examples

    def get_morph_category_enum_names(self):
        if self.morph_category_enum_names is None:
            self.load_data()
        return self.morph_category_enum_names

    def get_panel_enum_names(self):
        if self.panel_enum_names is None:
            self.load_data()
        return self.panel_enum_names

    def get_image_size(self):
        if self.image_size is None:
            self.load_data()
        return self.image_size

    def get_name_vector_size(self):
        if self.name_vector_size is None:
            self.load_data()
        return self.name_vector_size

    def __len__(self):
        return len(self.get_examples())

    def __getitem__(self, index):
        examples = self.get_examples()
        example = examples[index]
        rest_image = self.load_image(example["restImage"])
        morph_image = self.load_image(example["morphImage"])
        diff_image = self.load_image(example["diffImage"])
        name_vec = torch.tensor(example["name"], dtype=torch.float32)
        morph_category = self.get_morph_category_enum_names().index(example["morphCategoryEnumName"])
        panel_index = self.get_panel_enum_names().index(example["panelEnumName"])
        panel = torch.zeros(len(self.get_panel_enum_names()), dtype=torch.float32)
        panel[panel_index] = 1.0
        return [rest_image, morph_image, diff_image, name_vec, panel, morph_category]

    def load_image(self, file_name):
        if self.verbose:
            print("Loading %s ..." % file_name)
        with open(file_name, "rb") as file:
            image = extract_pytorch_image_from_filelike(file)
        if image.shape[1] != self.image_size or image.shape[2] != self.get_image_size():
            raise RuntimeError(file_name)
        return image


if __name__ == "__main__":
    dataset = MorphCategoryClassificationDataset("data/bougain/_20200401/morphtags_training_data/training.json")
    examples = dataset.get_examples()
    print(len(examples))
