from pytasuku import Workspace
import os
import urllib.request
import gzip
import codecs
import numpy
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pytasuku.util import create_delete_all_task


class Constants:
    DIR = "data/mnist"
    RAW_DIR = DIR + "/raw"

    URLS = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    GZ_FILES = [
        RAW_DIR + "/train-images-idx3-ubyte.gz",
        RAW_DIR + "/train-labels-idx1-ubyte.gz",
        RAW_DIR + "/t10k-images-idx3-ubyte.gz",
        RAW_DIR + "/t10k-labels-idx1-ubyte.gz",
    ]

    TRAINING_IMAGE_RAW_FILE = RAW_DIR + "/train-images-idx3-ubyte"
    TRAINING_LABEL_RAW_FILE = RAW_DIR + "/train-labels-idx1-ubyte"
    TEST_IMAGE_RAW_FILE = RAW_DIR + "/t10k-images-idx3-ubyte"
    TEST_LABEL_RAW_FILE = RAW_DIR + "/t10k-labels-idx1-ubyte"

    RAW_FILES = [
        TRAINING_IMAGE_RAW_FILE,
        TRAINING_LABEL_RAW_FILE,
        TEST_IMAGE_RAW_FILE,
        TEST_LABEL_RAW_FILE,
    ]

    TRAINING_FILE = DIR + "/training.pt"
    TEST_FILE = DIR + "/test.pt"

    SAMPLE_IMAGES_FILE = DIR + "/sample_images.png"


# Most of the code below is taken and modified from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

def download_mnist():
    os.makedirs(Constants.RAW_DIR, exist_ok=True)
    for url in Constants.URLS:
        file_name = url.split('/')[-1]
        urllib.request.urlretrieve(
            url,
            Constants.RAW_DIR + "/" + file_name)


def uncompress_mnist():
    for gz_file in Constants.GZ_FILES:
        with open(gz_file.replace('.gz', ''), 'wb') as out_f, gzip.GzipFile(gz_file) as zip_f:
            out_f.write(zip_f.read())


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def write_training_file():
    training_images = read_image_file(Constants.TRAINING_IMAGE_RAW_FILE)
    training_labels = read_label_file(Constants.TRAINING_LABEL_RAW_FILE)
    with open(Constants.TRAINING_FILE, 'wb') as f:
        torch.save((training_images, training_labels), f)


def write_test_file():
    test_images = read_image_file(Constants.TEST_IMAGE_RAW_FILE)
    test_labels = read_label_file(Constants.TEST_LABEL_RAW_FILE)
    with open(Constants.TEST_FILE, 'wb') as f:
        torch.save((test_images, test_labels), f)


def prepare_4x4_images(numpy_images):
    plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.5)

    for i in range(16):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(numpy_images[i, :, :], cmap='gray')


def show_images():
    images, labels = torch.load(Constants.TRAINING_FILE)
    numpy_images = images[:16, :, :].numpy()
    prepare_4x4_images(numpy_images)
    plt.show()


def sample_images():
    images, labels = torch.load(Constants.TRAINING_FILE)
    numpy_images = images[:16, :, :].numpy()
    prepare_4x4_images(numpy_images)
    plt.savefig(Constants.SAMPLE_IMAGES_FILE, format='png')


def define_tasks(workspace: Workspace):
    for gz_file in Constants.GZ_FILES:
        workspace.create_file_task(gz_file, [], download_mnist)
    workspace.create_command_task(Constants.DIR + "/download", Constants.GZ_FILES)

    for raw_file in Constants.RAW_FILES:
        workspace.create_file_task(raw_file, Constants.GZ_FILES, uncompress_mnist)
    workspace.create_command_task(Constants.DIR + "/uncompress", Constants.RAW_FILES)

    workspace.create_file_task(Constants.TRAINING_FILE, Constants.RAW_FILES, write_training_file)
    workspace.create_file_task(Constants.TEST_FILE, Constants.RAW_FILES, write_test_file)
    workspace.create_command_task(Constants.DIR + "/process", [Constants.TRAINING_FILE, Constants.TEST_FILE])

    create_delete_all_task(workspace, Constants.DIR + "/clean", Constants.RAW_FILES + Constants.GZ_FILES +
                           [Constants.TRAINING_FILE, Constants.TEST_FILE])

    workspace.create_command_task(Constants.DIR + "/show_images", [Constants.TRAINING_FILE], show_images)

    workspace.create_file_task(Constants.SAMPLE_IMAGES_FILE, [Constants.TRAINING_FILE], sample_images)
