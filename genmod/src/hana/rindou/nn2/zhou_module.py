from typing import List

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, ReLU, InstanceNorm2d, ConvTranspose2d, Tanh

from hana.rindou.nn2.conv import DownsampleBlock, UpsampleBlock
from hana.rindou.nn2.init_function import create_init_function
from hana.rindou.nn2.linear import LinearBlock
from hana.rindou.nn2.view_change import ViewImageAsVector, ViewVectorAsMultiChannelImage, ViewChange


class ZhouModule(Module):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 initial_channels: int = 64,
                 image_repr_size: int = 4096,
                 pose_repr_sizes: List[int] = None,
                 max_channel: int = 512,
                 initialization_method: str = 'he'):
        super().__init__()
        init = create_init_function(initialization_method)

        if pose_repr_sizes is None:
            pose_repr_sizes = [128, 256]

        downsampling_layers = []
        upsampling_layers = []

        downsampling_layers.append(DownsampleBlock(image_channels, initial_channels, initialization_method))
        upsampling_layers.insert(0, Sequential(
            UpsampleBlock(initial_channels, initial_channels, initialization_method),
            Conv2d(in_channels=initial_channels, out_channels=2, kernel_size=1, stride=1, padding=0),
            Tanh()))

        element_count = (image_size // 2) * (image_size // 2) * initial_channels
        current_channel_count = initial_channels
        current_image_size = image_size // 2
        while current_image_size > 1 and element_count > image_repr_size:
            next_channel_count = min(current_channel_count * 2, max_channel)
            next_element_count = next_channel_count * (current_image_size ** 2) // 4
            if next_element_count < image_repr_size:
                break

            downsampling_layers.append(
                DownsampleBlock(current_channel_count, next_channel_count, initialization_method))
            upsampling_layers.insert(
                0,
                UpsampleBlock(next_channel_count, current_channel_count, initialization_method))
            current_image_size //= 2
            current_channel_count = next_channel_count
            element_count = (current_image_size ** 2) * current_channel_count

        if element_count > image_repr_size:
            new_channel_count = current_channel_count * image_repr_size // element_count
            downsampling_layers.append(
                Sequential(
                    init(Conv2d(in_channels=current_channel_count,
                                out_channels=new_channel_count,
                                kernel_size=1,
                                padding=0,
                                stride=1)),
                    InstanceNorm2d(new_channel_count, affine=True),
                    ReLU(inplace=True)))
            upsampling_layers.insert(
                0,
                Sequential(
                    init(Conv2d(in_channels=new_channel_count,
                                out_channels=current_channel_count,
                                kernel_size=1,
                                padding=0,
                                stride=1)),
                    InstanceNorm2d(current_channel_count, affine=True),
                    ReLU(inplace=True)))
            downsampling_layers.append(ViewImageAsVector())
            upsampling_layers.insert(0, ViewChange([new_channel_count, current_image_size, current_image_size]))
        else:
            downsampling_layers.append(ViewImageAsVector())
            upsampling_layers.insert(0, ViewChange([current_channel_count, current_image_size, current_image_size]))

        self.downsampler = Sequential(*downsampling_layers)
        self.upsampler = Sequential(*upsampling_layers)

        pose_layers = []
        pose_sizes = [pose_size] + pose_repr_sizes
        for i in range(len(pose_sizes) - 1):
            pose_layers.append(
                LinearBlock(in_features=pose_sizes[i],
                            out_features=pose_sizes[i + 1],
                            initialization_method=initialization_method))
        self.pose_xform = Sequential(*pose_layers)

        self.image_pose_combiner = Sequential(
            LinearBlock(
                in_features=pose_repr_sizes[-1] + image_repr_size,
                out_features=image_repr_size,
                initialization_method=initialization_method),
            LinearBlock(
                in_features=image_repr_size,
                out_features=image_repr_size,
                initialization_method=initialization_method))

    def forward(self, image: Tensor, pose: Tensor):
        """
        image is of size [n, c, w, h].
        pose is of size [n, p] where p is the size of the pose vector.
        The method returns a tensor of size [n, 2, w, h], which needs to be converted by the user of the class to
          a correct appearance flow tensor.
        """
        image_repr = self.downsampler(image)
        pose_repr = self.pose_xform(pose)
        combined_repr = torch.cat([image_repr, pose_repr], dim=1)
        repr = self.image_pose_combiner(combined_repr)
        return self.upsampler(repr)


if __name__ == "__main__":
    cuda = torch.device("cuda")
    module = ZhouModule().to(cuda)

    image = torch.zeros(64, 4, 256, 256, device=cuda)
    pose = torch.zeros(64, 3, device=cuda)
    output = module(image, pose)
    print(output.shape)
