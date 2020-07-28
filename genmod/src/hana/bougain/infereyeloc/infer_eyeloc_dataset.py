import json

import math

import PIL.Image
import numpy
import torch
from torch.utils.data import Dataset

from hana.rindou.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image


class InferEyeLocDataset(Dataset):
    def __init__(self,
                 data_json_file_name: str,
                 verbose: bool = False):
        self.verbose = verbose
        self.data_json_file_name = data_json_file_name

        self.examples = None
        self.image_size = None

    def load_data(self):
        with open(self.data_json_file_name, "rt", encoding="utf-8") as json_file:
            data = json.load(json_file)
        self.examples = data["examples"]
        self.image_size = data["image_size"]

    def get_examples(self):
        if self.examples is None:
            self.load_data()
        return self.examples

    def get_image_size(self):
        if self.image_size is None:
            self.load_data()
        return self.image_size

    def __len__(self):
        return len(self.get_examples())

    def __getitem__(self, index):
        examples = self.get_examples()
        example = examples[index]
        image = self.load_image(example[0])
        eye_spec = torch.tensor([example[1], example[2], example[3], example[4]])
        return [image, eye_spec]

    def load_image(self, file_name):
        if self.verbose:
            print("Loading %s ..." % file_name)
        with open(file_name, "rb") as file:
            image = extract_pytorch_image_from_filelike(file)
        if image.shape[1] != self.image_size or image.shape[2] != self.get_image_size():
            raise RuntimeError(file_name)
        return image

if __name__ == "__main__":
    dataset = InferEyeLocDataset("data/bougain/_20200502/infereyeloc_training_data/training.json")
    examples = dataset.get_examples()
    print(len(examples))

    image_size = dataset.get_image_size()
    example = dataset[0]
    image = example[0]
    left = int(math.floor(example[1][0].item() * image_size))
    right = int(math.floor(example[1][1].item() * image_size))
    bottom = int(math.floor((1.0 - example[1][2].item()) * image_size))
    top = int(math.floor((1 - example[1][3].item()) * image_size))

    for i in range(image_size):
        image[0, bottom, i] = 1.0
        image[1, bottom, i] = -1.0
        image[2, bottom, i] = -1.0
        image[3, bottom, i] = 1.0

        image[0, top, i] = -1.0
        image[1, top, i] = 1.0
        image[2, top, i] = -1.0
        image[3, top, i] = 1.0

    numpy_image = rgba_to_numpy_image(image)
    pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
    pil_image.save("data/bougain/_20200502/infereyeloc_training_data/tests.png")
