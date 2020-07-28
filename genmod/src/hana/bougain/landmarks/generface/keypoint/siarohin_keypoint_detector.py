from typing import List

import torch
from torch.nn import Module, Conv2d
from torch.nn.functional import softplus

from hana.bougain.nn.common import heatmap_to_keypoint
from hana.bougain.nn.siarohin_modules import SiarohinUNet, SiarohinSameSizeBlock
from hana.rindou.nn2.init_function import create_init_function


# Code inspired by https://github.com/AliaksandrSiarohin/first-order-model/blob/master/modules/keypoint_detector.py
class SiarohinKeypointDetector(Module):
    def __init__(self,
                 num_keypoints: int = 32,
                 in_channels: int = 3,
                 temperature: float = 0.1,
                 hidden_channels: int = 32,
                 num_blocks=5,
                 max_channels=1024,
                 activation: str = 'relu'):
        super().__init__()
        init = create_init_function('he')
        self.temperature = temperature
        self.num_keypoints = num_keypoints

        self.first = SiarohinSameSizeBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation)
        self.u_net = SiarohinUNet(
            in_channels=hidden_channels,
            num_blocks=num_blocks,
            max_channels=max_channels,
            activation=activation)
        self.last = init(Conv2d(
            in_channels=hidden_channels,
            out_channels=num_keypoints,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        n, c, h, w = x.shape
        score = self.last(self.u_net(self.first(x))).view(n, self.num_keypoints, h * w)
        score = softplus(score)
        heatmap = (score / score.sum(dim=2, keepdim=True)).view(n, self.num_keypoints, h, w)
        keypoints = heatmap_to_keypoint(heatmap)
        return [keypoints, heatmap]


if __name__ == "__main__":
    cuda = torch.device("cuda")
    heatmap = torch.softmax(torch.zeros(16, 10, 256, 256, device=cuda).view(16, 10, 256 * 256), dim=2) \
        .view(16, 10, 256, 256)
    print(heatmap)
    output = heatmap_to_keypoint(heatmap)
    print(output.shape)
    # print(output)

    print(torch.softmax(torch.tensor([0.0, 0.0, 0.0]), dim=0))
