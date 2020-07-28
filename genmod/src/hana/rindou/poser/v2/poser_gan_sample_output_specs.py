import os

import PIL
import numpy
from PIL import Image
from torch.nn import Module
from torch.nn.functional import interpolate

from hana.rindou.poser.v1.poser_gan_tasks_ver2 import PoserGanSampleOutputSpecVer2
from hana.rindou.util import rgba_to_numpy_image_greenscreen


class SourceTargetGeneratedSampleOutputSpec(PoserGanSampleOutputSpecVer2):
    def __init__(self,
                 count: int = 12,
                 example_per_row: int = 3,
                 example_per_sample_count: int = 2000,
                 sample_output_index_seed: int = 147258399,
                 image_size: int = 256):
        super().__init__()
        self._count = count
        self._example_per_row = example_per_row
        self._example_per_sample_output = example_per_sample_count
        self._sample_output_index_seed = sample_output_index_seed
        self._image_size = image_size

    @property
    def count(self) -> int:
        return self._count

    @property
    def example_per_row(self) -> int:
        return self._example_per_row

    @property
    def example_per_sample_output(self) -> int:
        return self._example_per_sample_output

    @property
    def sample_output_index_seed(self) -> int:
        return self._sample_output_index_seed

    @property
    def image_size(self) -> int:
        return self._image_size


class Image01TargetGeneratedSampleOutputSpec(PoserGanSampleOutputSpecVer2):
    def __init__(self,
                 count: int = 8,
                 example_per_row: int = 2,
                 example_per_sample_count: int = 2000,
                 sample_output_index_seed: int = 846218913,
                 image_size: int = 256):
        super().__init__()
        self._count = count
        self._example_per_row = example_per_row
        self._example_per_sample_output = example_per_sample_count
        self._sample_output_index_seed = sample_output_index_seed
        self._image_size = image_size

    @property
    def count(self) -> int:
        return self._count

    @property
    def example_per_row(self) -> int:
        return self._example_per_row

    @property
    def example_per_sample_output(self) -> int:
        return self._example_per_sample_output

    @property
    def sample_output_index_seed(self) -> int:
        return self._sample_output_index_seed

    @property
    def image_size(self) -> int:
        return self._image_size

    def save_sample_image(self, G: Module, sample_output_batch, file_name: str):
        G.train(False)
        image_0 = sample_output_batch[0]
        image_1 = sample_output_batch[1]
        pose = sample_output_batch[2]
        target_image = sample_output_batch[-1]

        output = G(image_0, image_1, pose)
        self.save_output_image_2(image_0, image_1, target_image, output[0].detach(), file_name)

    def save_output_image_2(self, images_0, images_1, target_images, output_images, file_name):
        images_0 = interpolate(images_0, size=self.image_size).cpu()
        images_1 = interpolate(images_1, size=self.image_size).cpu()
        target_images = interpolate(target_images, size=self.image_size).cpu()
        output_images = interpolate(output_images, size=self.image_size).cpu()

        n = output_images.shape[0]
        num_rows = n // self.example_per_row
        if n % self.example_per_row != 0:
            num_rows += 1
        num_cols = 4 * self.example_per_row

        image_size = self.image_size
        output = numpy.zeros([num_rows * image_size, num_cols * image_size, 3])
        for i in range(n):
            row = i // self.example_per_row
            col = i % self.example_per_row

            image_0 = rgba_to_numpy_image_greenscreen(images_0[i])
            image_1 = rgba_to_numpy_image_greenscreen(images_1[i])
            target_image = rgba_to_numpy_image_greenscreen(target_images[i])
            output_image = rgba_to_numpy_image_greenscreen(output_images[i])

            row_start = row * image_size
            row_end = row_start + image_size

            col_start = (4 * col) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = image_0

            col_start = (4 * col + 1) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = image_1

            col_start = (4 * col + 2) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = target_image

            col_start = (4 * col + 3) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = output_image

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pil_image = Image.fromarray(numpy.uint8(numpy.rint(output * 255.0)), mode='RGB')
        pil_image.save(file_name)
        print("Saved %s" % file_name)


class SourceTargetTwoOutputsSampleOutputSpec(PoserGanSampleOutputSpecVer2):
    def __init__(self):
        super().__init__()

    @property
    def count(self) -> int:
        return 8

    @property
    def example_per_row(self) -> int:
        return 2

    @property
    def example_per_sample_output(self) -> int:
        return 2000

    @property
    def sample_output_index_seed(self) -> int:
        return 147258399

    @property
    def image_size(self) -> int:
        return 256

    def save_sample_image(self, G: Module, sample_output_batch, file_name: str):
        G.train(False)
        source_image = sample_output_batch[0]
        pose = sample_output_batch[1]
        target_image = sample_output_batch[2]

        output = G(source_image, pose)
        self.save_output_image_2(source_image, target_image, output[0].detach(), output[1].detach(), file_name)

    def save_output_image_2(self, source_images, target_images, first_output_images, second_output_images, file_name):
        source_images = interpolate(source_images, size=self.image_size).cpu()
        target_images = interpolate(target_images, size=self.image_size).cpu()
        first_output_images = interpolate(first_output_images, size=self.image_size).cpu()
        second_output_images = interpolate(second_output_images, size=self.image_size).cpu()

        n = first_output_images.shape[0]
        num_rows = n // self.example_per_row
        if n % self.example_per_row != 0:
            num_rows += 1
        num_cols = 4 * self.example_per_row

        image_size = self.image_size
        output = numpy.zeros([num_rows * image_size, num_cols * image_size, 3])
        for i in range(n):
            row = i // self.example_per_row
            col = i % self.example_per_row

            source_image = rgba_to_numpy_image_greenscreen(source_images[i])
            target_image = rgba_to_numpy_image_greenscreen(target_images[i])
            first_output_image = rgba_to_numpy_image_greenscreen(first_output_images[i])
            second_output_image = rgba_to_numpy_image_greenscreen(second_output_images[i])

            row_start = row * image_size
            row_end = row_start + image_size

            col_start = (4 * col) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = source_image

            col_start = (4 * col + 1) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = target_image

            col_start = (4 * col + 2) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = first_output_image

            col_start = (4 * col + 3) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = second_output_image

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(output * 255.0)), mode='RGB')
        pil_image.save(file_name)
        print("Saved %s" % file_name)
