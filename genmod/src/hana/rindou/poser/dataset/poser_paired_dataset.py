import csv

import torch
from torch.utils.data import Dataset

from hana.rindou.util import extract_pytorch_image_from_filelike


class PoserPairedDataset(Dataset):
    def __init__(self, paired_tsv_file_name: str, image_size: int = 256, verbose=False):
        self.paired_tsv_file_name = paired_tsv_file_name
        self.examples = None
        self.image_size = image_size
        self.verbose = verbose

    def get_examples(self):
        if self.examples is None:
            examples = load_paired_tsv_file(self.paired_tsv_file_name)
            self.examples = examples
        return self.examples

    def __len__(self):
        return len(self.get_examples())

    def __getitem__(self, index):
        examples = self.get_examples()
        example = examples[index]
        source_image = self.load_image(example[0])
        pose = torch.Tensor(example[1])
        target_image = self.load_image(example[2])
        return source_image, pose, target_image

    def load_image(self, file_name):
        if self.verbose:
            print("Loading %s ..." % file_name)
        with open(file_name, "rb") as file:
            return extract_pytorch_image_from_filelike(file)


def load_paired_tsv_file(paired_tsv_file_name):
    examples = []
    paired_tsv_file = open(paired_tsv_file_name)
    tsvreader = csv.reader(paired_tsv_file, delimiter='\t')
    for line in tsvreader:
        example = []
        example.append(line[0])
        pose = [float(x) for x in line[1:-1]]
        example.append(pose)
        example.append(line[-1])
        examples.append(example)
    paired_tsv_file.close()
    return examples


if __name__ == "__main__":
    paired_tsv_file_name = "data/rindou/_20190821/close_up_256/train/paired.tsv"
    dataset = PoserPairedDataset(paired_tsv_file_name)
    loaded = set()
    count = 0
    examples = dataset.get_examples()
    total = len(examples)
    for example in examples:
        if example[0] not in loaded:
            dataset.load_image(example[0])
            loaded.add(example[0])
        dataset.load_image(example[2])
        count += 1
        if count % 1000 == 0:
            print("%d out of %d" % (count, total))