import os

import numpy
import torch
from matplotlib import pyplot as plt, gridspec as gridspec
from torch.nn import functional as F


def is_power2(x):
    return x != 0 and ((x & (x - 1)) == 0)


def torch_save(content, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        torch.save(content, f)


def torch_load(file_name):
    with open(file_name, 'rb') as f:
        return torch.load(f)


def srgb_to_linear(x):
    x = numpy.clip(x, 0.0, 1.0)
    x_low = (x < 0.4045).astype(float)
    return x_low * x / 12.92 + (1 - x_low) * ((x + 0.055) / 1.055) ** 2.4


def linear_to_srgb(x):
    x = numpy.clip(x, 0.0, 1.0)
    x_low = (x < 0.0031308).astype(float)
    return x_low * x * 12.92 + (1 - x_low) * ((1 + 0.055) * (x ** (1.0 / 2.4)) - 0.055)


def save_sample_images(sample_images: torch.Tensor,
                       sample_image_size: int,
                       sample_images_per_row: int,
                       file_name: str):
    n = sample_images.shape[0]
    scale_factor = sample_image_size / sample_images.shape[2]
    sample_images = F.upsample(sample_images, scale_factor=scale_factor)

    num_rows = n // sample_images_per_row
    if n % sample_images_per_row != 0:
        num_rows += 1
    plt.figure(figsize=(num_rows, sample_images_per_row))
    gs = gridspec.GridSpec(num_rows, sample_images_per_row)

    for i in range(n):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample_image_linear = (sample_images[i].numpy()
                               .reshape(3, sample_image_size * sample_image_size)
                               .transpose()
                               .reshape(sample_image_size, sample_image_size, 3) + 1.0) / 2.0
        image = linear_to_srgb(sample_image_linear)
        plt.imshow(image)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name, format="png")
    plt.close()


def save_rng_state(file_name):
    rng_state = torch.get_rng_state()
    torch_save(rng_state, file_name)


def load_rng_state(file_name):
    rng_state = torch_load(file_name)
    torch.set_rng_state(rng_state)


def optimizer_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
