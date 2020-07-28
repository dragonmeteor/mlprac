import io
import os
import random
from collections import Iterable
from typing import Dict, Iterator, List

import PIL.Image
import torch
from torch import Tensor

from hana.rindou.poser.dataset.poser_paired_dataset import load_paired_tsv_file
from hana.rindou.util import torch_save, torch_load, rgba_to_numpy_image_greenscreen, extract_pytorch_image_from_filelike
from pytasuku import Workspace
from pytasuku.workspace import do_nothing


class PoserPairedChunkedPicklesTasks:
    def __init__(self,
                 workspace: Workspace,
                 dir: str,
                 paired_tsv_file_name: str,
                 num_chunks: int,
                 image_size: int = 256,
                 random_seed: int = 114514):
        self.workspace = workspace
        self.dir = dir
        self.paired_tsv_file_name = paired_tsv_file_name
        self.num_chunks = num_chunks
        self.image_size = image_size
        self.random_seed = random_seed

        self._examples_file_name_list = [self.examples_file_name(i) for i in range(self.num_chunks)]
        self.define_tasks()

        self.rest_images = None

    def rest_images_file_name(self):
        return self.dir + "/rest_images.pt"

    def examples_file_name(self, index):
        return self.dir + ("/examples_%08d.pt" % index)

    @property
    def examples_file_name_list(self):
        return self._examples_file_name_list

    def examples_done_file_name(self):
        return self.dir + "/examples_done.txt"

    def define_tasks(self):
        self.workspace.create_file_task(self.rest_images_file_name(), [], lambda: self.create_rest_images_file())
        self.workspace.create_file_task(self.examples_done_file_name(), [], lambda: self.create_examples_files())
        self.workspace.create_command_task(self.dir + "/create",
                                           [self.rest_images_file_name(), self.examples_done_file_name()],
                                           do_nothing)

    def create_examples_files(self):
        examples = load_paired_tsv_file(self.paired_tsv_file_name)
        rest_images_file_name_to_index = self.extract_rest_images_file_name_to_index(examples)

        m = len(examples) // self.num_chunks
        remainder = len(examples) % self.num_chunks
        random.seed(self.random_seed)
        permuted_examples = random.sample(examples, len(examples))
        chunks = [permuted_examples[i * m:(i + 1) * m] for i in range(self.num_chunks)]
        for i in range(remainder):
            chunks[i].append(permuted_examples[self.num_chunks * m + i])

        for i in range(self.num_chunks):
            self.create_examples_file(i, chunks[i], rest_images_file_name_to_index)

        os.makedirs(os.path.dirname(self.examples_done_file_name()), exist_ok=True)
        with open(self.examples_done_file_name(), "wt") as f:
            f.write("DONE!!!")

    def create_examples_file(self, index: int, chunk, rest_images_file_name_to_index: Dict[str, int]):
        output = []
        total = len(chunk)
        count = 0
        for example in chunk:
            count += 1
            print("Loading %s ... (%d out of %d)" % (example[2], count, total))
            record = [
                rest_images_file_name_to_index[example[0]],
                torch.Tensor(example[1]),
                self.read_image_file(example[2])
            ]
            output.append(record)
        file_name = self.examples_file_name(index)
        print("Saving %s ..." % file_name)
        torch_save(output, file_name)

    @staticmethod
    def extract_rest_images_file_name_to_index(examples) -> Dict[str, int]:
        rest_images_file_name_to_index = {}
        count = 0
        for example in examples:
            image_file_name = example[0]
            if image_file_name not in rest_images_file_name_to_index:
                rest_images_file_name_to_index[image_file_name] = count
                count += 1
        return rest_images_file_name_to_index

    def read_image_file(self, image_file_name):
        with open(image_file_name, mode='rb') as file:
            content = file.read()
            image = PIL.Image.open(io.BytesIO(content))
            assert image.width == self.image_size
            assert image.height == self.image_size
            print("Reading", image_file_name, "...")
        return content

    def create_rest_images_file(self):
        examples = load_paired_tsv_file(self.paired_tsv_file_name)
        rest_images_file_name_to_index = self.extract_rest_images_file_name_to_index(examples)

        output = [None for i in range(len(rest_images_file_name_to_index))]
        count = 0
        total = len(rest_images_file_name_to_index)
        for image_file_name in rest_images_file_name_to_index:
            index = rest_images_file_name_to_index[image_file_name]
            content = self.read_image_file(image_file_name)
            output[index] = content
            count += 1
            print("Done! (Process %d out of %d)" % (count, total))

        torch_save(output, self.rest_images_file_name())

    def get_rest_images(self):
        if self.rest_images is None:
            self.rest_images = torch_load(self.rest_images_file_name())
        return self.rest_images

    def get_chuck(self, index):
        return torch_load(self.examples_file_name(index))

    def create_data_loader(self, batch_size):
        return PoserPairedChunkedPicklesDataLoader(self, batch_size)


class PoserPairedChunkedPicklesDataLoader(Iterable):
    def __init__(self, tasks: PoserPairedChunkedPicklesTasks, batch_size: int):
        self.tasks = tasks
        self.batch_size = batch_size
        self.rest_images = None

    def __iter__(self) -> Iterator[List[Tensor]]:
        return PoserPairedChuckedPicklesIterator(self.tasks, self.batch_size)


class PoserPairedChuckedPicklesIterator(Iterator[List[Tensor]]):
    def __init__(self, tasks: PoserPairedChunkedPicklesTasks, batch_size: int):
        self.tasks = tasks
        self.batch_size = batch_size
        self.chunk_indices = torch.randperm(self.tasks.num_chunks).numpy().tolist()
        self.chunk_item_indices = []
        self.chunk = None

    def __next__(self) -> List[Tensor]:
        count = 0
        source_image_list = []
        post_list = []
        target_image_list = []

        while count < self.batch_size:
            if self.no_more_data():
                raise StopIteration
            self.maybe_load_neck_chunk()
            limit = min(self.batch_size - count, len(self.chunk_item_indices))
            for i in range(limit):
                example_index = self.chunk_item_indices.pop()
                example = self.chunk[example_index]
                source_image_list.append(self.load_source_image(example[0]))
                post_list.append(example[1])
                target_image_list.append(extract_pytorch_image_from_filelike(io.BytesIO(example[2])))
                count += 1

        source_image = torch.cat([x.unsqueeze(0) for x in source_image_list], dim=0)
        pose = torch.cat([x.unsqueeze(0) for x in post_list], dim=0)
        target_image = torch.cat([x.unsqueeze(0) for x in target_image_list], dim=0)
        return [source_image, pose, target_image]

    def no_more_data(self) -> bool:
        return len(self.chunk_indices) == 0 and len(self.chunk_item_indices) < self.batch_size

    def maybe_load_neck_chunk(self):
        if len(self.chunk_item_indices) == 0:
            chunk_index = self.chunk_indices.pop()
            self.chunk = self.tasks.get_chuck(chunk_index)
            self.chunk_item_indices = torch.randperm(len(self.chunk)).numpy().tolist()

    def load_source_image(self, index):
        content = self.tasks.get_rest_images()[index]
        return extract_pytorch_image_from_filelike(io.BytesIO(content))


if __name__ == "__main__":
    workspace = Workspace()
    tasks = PoserPairedChunkedPicklesTasks(
        workspace=workspace,
        dir="data/rindou/_20190824/close_up_256/validation",
        paired_tsv_file_name="data/rindou/_20190821/close_up_256/validation/paired.tsv",
        num_chunks=10)
    data_loader = tasks.create_data_loader(16)
    iterator = iter(data_loader)
    output = next(iterator)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)

    import scipy.misc
    scipy.misc.imsave("source.png", rgba_to_numpy_image_greenscreen(output[0][0]))
    scipy.misc.imsave("target.png", rgba_to_numpy_image_greenscreen(output[2][0]))