from typing import List

from torch.nn import Sequential
from torchvision.models import vgg16

VGG16_LAYERS = {
    "conv1_1": 0,
    "relu1_1": 1,
    "conv1_2": 2,
    "relu1_2": 3,
    "pool1": 4,
    "conv2_1": 5,
    "relu2_1": 6,
    "conv2_2": 7,
    "relu2_2": 8,
    "pool2": 9,
    "conv3_1": 10,
    "relu3_1": 11,
    "conv3_2": 12,
    "relu3_2": 13,
    "conv3_3": 14,
    "relu3_3": 15,
    "pool3": 16,
    "conv4_1": 17,
    "relu4_1": 18,
    "conv4_2": 19,
    "relu4_2": 20,
    "conv4_3": 21,
    "relu4_3": 22,
    "pool4": 23,
    "conv5_1": 24,
    "relu5_1": 25,
    "conv5_2": 26,
    "relu5_2": 27,
    "conv5_3": 28,
    "relu5_3": 29,
    "pool5": 30,
}


def get_vgg16_perceptual_loss_modules(layers: List[str]):
    for layer in layers:
        assert layer in VGG16_LAYERS
    layer_indices = sorted([VGG16_LAYERS[layer] + 1 for layer in layers])
    layer_indices.insert(0, 0)
    vgg16_features = vgg16(pretrained=True).features
    return [Sequential(vgg16_features[layer_indices[i]:layer_indices[i + 1]]) for i in range(len(layer_indices) - 1)]


if __name__ == "__main__":
    modules = get_vgg16_perceptual_loss_modules(["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
    print(modules)
