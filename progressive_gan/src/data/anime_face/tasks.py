import numpy
import torch
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F

from gans.util import torch_save, torch_load
from pytasuku import Workspace

IMAGE_COUNT = 21551

DIR = "data/anime_face"

SAMPLE_IMAGE_INDICES_FILE_NAME = DIR + "/sample_image_indices.pt"

SAMPLE_IMAGE_COUNT = 100
SAMPLE_IMAGES_PER_ROW = 10


def image_data_file_name(size):
    return DIR + "/image_data_%02dx%02d.pt" % (size, size)


def sample_images_file_name(size):
    return DIR + "/sample_images_%02dx%02d.png" % (size, size)


def srgb_to_linear(x):
    x = numpy.clip(x, 0.0, 1.0)
    x_low = (x < 0.4045).astype(float)
    return x_low * x / 12.92 + (1 - x_low) * ((x + 0.055) / 1.055) ** 2.4


def linear_to_srgb(x):
    x = numpy.clip(x, 0.0, 1.0)
    x_low = (x < 0.0031308).astype(float)
    return x_low * x * 12.92 + (1 - x_low) * ((1 + 0.055) * (x ** (1.0 / 2.4)) - 0.055)


def create_image_data_64x64():
    data = torch.zeros(IMAGE_COUNT, 3, 64, 64)
    for image_index in range(IMAGE_COUNT):
        file_name = DIR + "/images/%d.png" % (image_index + 1)
        image = PIL.Image.open(file_name, "r")
        numpy_image = srgb_to_linear(numpy.asarray(image) / 255.0).reshape(64 * 64, 3).transpose().reshape(3, 64, 64)
        torch_image = torch.from_numpy(numpy_image) * 2.0 - 1.0
        data[image_index] = torch_image
        if (image_index + 1) % 100 == 0:
            print("Processed %d images ..." % (image_index + 1))
    torch_save(data, image_data_file_name(64))


def create_image_data(size: int):
    images64 = torch_load(image_data_file_name(64))
    down_images = F.avg_pool2d(images64, kernel_size=64 // size, stride=64 // size, padding=0)
    torch_save(down_images, image_data_file_name(size))


def create_sample_image_indices():
    random_indices = torch.randint(low=0, high=IMAGE_COUNT, size=(SAMPLE_IMAGE_COUNT,), dtype=torch.long, out=None)
    torch_save(random_indices, SAMPLE_IMAGE_INDICES_FILE_NAME)


def create_sample_images(size):
    images = torch_load(image_data_file_name(size))
    sample_image_indices = torch_load(SAMPLE_IMAGE_INDICES_FILE_NAME)
    n = sample_image_indices.shape[0]
    sample_images = images[sample_image_indices]
    scale_factor = 64 / images.shape[2]
    sample_images = F.upsample(sample_images, scale_factor=scale_factor)

    num_rows = n // SAMPLE_IMAGES_PER_ROW
    if n % SAMPLE_IMAGES_PER_ROW != 0:
        num_rows += 1
    plt.figure(figsize=(num_rows, SAMPLE_IMAGES_PER_ROW))
    gs = gridspec.GridSpec(num_rows, SAMPLE_IMAGES_PER_ROW)

    for i in range(n):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        image = linear_to_srgb(
            (sample_images[i].numpy().reshape(3, 64 * 64).transpose().reshape(64, 64, 3) + 1.0) / 2.0)
        plt.imshow(image)

    plt.savefig(sample_images_file_name(size), format="png")
    plt.close()


def define_tasks(workspace: Workspace):
    workspace.create_file_task(image_data_file_name(64), [], create_image_data_64x64)

    for size in [4, 8, 16, 32]:
        def create_func(size):
            def create_it():
                create_image_data(size)

            return create_it

        workspace.create_file_task(image_data_file_name(size), [image_data_file_name(64)], create_func(size))

    workspace.create_command_task(DIR + "/images_data",
                                  [image_data_file_name(size) for size in [4, 8, 16, 32, 64]])

    workspace.create_file_task(SAMPLE_IMAGE_INDICES_FILE_NAME, [], create_sample_image_indices)

    for size in [4, 8, 16, 32, 64]:
        def create_func(size):
            def create_it():
                create_sample_images(size)

            return create_it

        workspace.create_file_task(sample_images_file_name(size),
                                   [
                                       SAMPLE_IMAGE_INDICES_FILE_NAME,
                                       image_data_file_name(size)
                                   ],
                                   create_func(size))

    workspace.create_command_task(DIR + "/sample_images",
                                  [sample_images_file_name(size) for size in [4, 8, 16, 32, 64]])
