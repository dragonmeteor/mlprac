import os

import PIL.Image
import numpy
import torch
from torch import Tensor


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
    return numpy.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = numpy.clip(x, 0.0, 1.0)
    return numpy.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


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


def rgb_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    numpy_image = (torch_image.numpy().reshape(3, height * width).transpose().reshape(height, width,
                                                                                      3) - min_pixel_value) \
                  / (max_pixel_value - min_pixel_value)
    return linear_to_srgb(numpy_image)


def rgba_to_numpy_image_greenscreen(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    numpy_image = (torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width,
                                                                                      4) - min_pixel_value) \
                  / (max_pixel_value - min_pixel_value)
    rgb_image = linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgb_image[:, :, 0:3] = rgb_image[:, :, 0:3] * a_image.reshape(a_image.shape[0], a_image.shape[1], 1)
    rgb_image[:, :, 1] = rgb_image[:, :, 1] + (1 - a_image)

    return rgb_image


def rgba_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 4
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width, 4)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    rgb_image = linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgba_image = numpy.concatenate((rgb_image, a_image.reshape(height, width, 1)), axis=2)
    return rgba_image


def rgb_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 3
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(3, height * width).transpose().reshape(height, width, 3)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    return linear_to_srgb(numpy_image)


def extract_pytorch_image_from_filelike(file, has_alpha=True, scale=2.0, offset=-1.0):
    try:
        pil_image = PIL.Image.open(file)
    except Exception as e:
        raise RuntimeError(file)
    return extract_pytorch_image_from_PIL_image(pil_image, has_alpha, scale, offset)


def extract_pytorch_image_from_PIL_image(pil_image, has_alpha=True, scale=2.0, offset=-1.0):
    if has_alpha:
        num_channel = 4
    else:
        num_channel = 3
    image_size = pil_image.width
    image = (numpy.asarray(pil_image) / 255.0).reshape(image_size, image_size, num_channel)
    image[:, :, 0:3] = srgb_to_linear(image[:, :, 0:3])
    image = image \
        .reshape(image_size * image_size, num_channel) \
        .transpose() \
        .reshape(num_channel, image_size, image_size)
    torch_image = torch.from_numpy(image).float() * scale + offset
    return torch_image


def extract_numpy_image_from_filelike(file):
    pil_image = PIL.Image.open(file)
    image_size = pil_image.width
    image = (numpy.asarray(pil_image) / 255.0).reshape(image_size, image_size, 4)
    image[:, :, 0:3] = srgb_to_linear(image[:, :, 0:3])
    return image


def convert_avs_to_avi(avs_file, avi_file):
    os.makedirs(os.path.dirname(avi_file), exist_ok=True)

    file = open("temp.vdub", "w")
    file.write("VirtualDub.Open(\"%s\");" % avs_file)
    file.write("VirtualDub.video.SetCompression(\"cvid\", 0, 10000, 0);")
    file.write("VirtualDub.SaveAVI(\"%s\");" % avi_file)
    file.write("VirtualDub.Close();")
    file.close()

    os.system("C:\\ProgramData\\chocolatey\\lib\\virtualdub\\tools\\vdub64.exe /i temp.vdub")

    os.remove("temp.vdub")


def convert_avi_to_mp4(avi_file, mp4_file):
    os.makedirs(os.path.dirname(mp4_file), exist_ok=True)
    os.system("ffmpeg -y -i %s -c:v libx264 -preset slow -crf 22 -c:a libfaac -b:a 128k %s" % \
              (avi_file, mp4_file))


def convert_avi_to_webm(avi_file, webm_file):
    os.makedirs(os.path.dirname(webm_file), exist_ok=True)
    os.system("ffmpeg -y -i %s -vcodec libvpx -qmin 0 -qmax 50 -crf 10 -b:v 1M -acodec libvorbis %s" % \
              (avi_file, webm_file))


def create_parent_dir(file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
