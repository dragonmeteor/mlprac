import torch
from torch.nn import ReLU, LeakyReLU
from torch.nn.functional import affine_grid

from hana.rindou.util import rgb_to_numpy_image


def activation_module(activation: str = 'relu'):
    if activation == "relu":
        return ReLU(inplace=True)
    else:
        return LeakyReLU(inplace=True, negative_slope=0.2)


def heatmap_to_keypoint(heatmap: torch.Tensor) -> torch.Tensor:
    n, c, h, w = heatmap.shape
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).to(heatmap.device).unsqueeze(0).repeat(n, 1, 1)
    grid = affine_grid(identity, [n, c, h, w], align_corners=False).view(n, 1, h * w, 2).repeat(1, c, 1, 1)
    return (heatmap.view(n, c, h * w, 1).repeat(1, 1, 1, 2) * grid).sum(dim=2)


def keypoint_to_gaussian(keypoint: torch.Tensor, image_size: int, variance: float) -> torch.Tensor:
    n, c, _ = keypoint.shape
    h, w = image_size, image_size
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).to(keypoint.device).unsqueeze(0).repeat(n, 1, 1)
    grid = affine_grid(identity, [n, c, h, w], align_corners=False).view(n, 1, h * w, 2).repeat(1, c, 1, 1)
    r2 = ((keypoint.view(n, c, 1, 2).repeat(1, 1, h * w, 1) - grid) ** 2).sum(dim=3)
    return torch.exp(-r2 / variance).view(n, c, h, w)


if __name__ == "__main__":
    from matplotlib import pyplot
    cuda = torch.device("cuda")
    keypoints = torch.zeros(1, 10, 2, device=cuda)
    gaussian = keypoint_to_gaussian(keypoints, image_size=256, variance=0.01)
    reconstructed_keypoints = heatmap_to_keypoint(gaussian)
    print(reconstructed_keypoints.shape)
    print(reconstructed_keypoints)

    numpy_image = rgb_to_numpy_image(gaussian[0,0:1,:,:].repeat(3,1,1).cpu(), min_pixel_value=0, max_pixel_value=1)
    pyplot.imshow(numpy_image)
    pyplot.show()