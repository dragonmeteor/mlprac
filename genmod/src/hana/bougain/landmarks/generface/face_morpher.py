from typing import List

import torch
from torch.nn import Module, Sequential, Sigmoid, Tanh

from hana.bougain.landmarks.generface.face_decomposer_and_morpher import add_list, one_by_one_conv, concat_list
from hana.bougain.nn.common import keypoint_to_gaussian
from hana.bougain.nn.residual_block import ResidualBlock
from hana.bougain.nn.siarohin_modules import SiarohinSameSizeBlock, SiarohinUNetEncoder, SiarohinUNetDecoder


class FaceMorpher(Module):
    def __init__(self,
                 in_channels: int = 3,
                 decomposer_in_channels: int = 32,
                 num_keypoints: int = 32,
                 num_blocks: int = 5,
                 max_channels: int = 1024,
                 keypoint_variance=0.01,
                 activation: str = 'relu'):
        super().__init__()
        self.keypoint_variance = keypoint_variance
        assert in_channels == 3 or in_channels == 4
        self.in_channels = in_channels

        self.source_image_first = SiarohinSameSizeBlock(
            in_channels=in_channels,
            out_channels=decomposer_in_channels,
            activation=activation)

        self.decomposer_encoder = SiarohinUNetEncoder(
            in_channels=decomposer_in_channels,
            num_blocks=num_blocks,
            max_channels=max_channels,
            activation=activation)

        image_bottleneck_channels = self.decomposer_encoder.get_num_outout_channels()[-1]

        self.keypoints_first = SiarohinSameSizeBlock(
            in_channels=num_keypoints,
            out_channels=decomposer_in_channels,
            activation=activation)

        self.keypoints_encoder = SiarohinUNetEncoder(
            in_channels=decomposer_in_channels,
            num_blocks=num_blocks,
            max_channels=max_channels,
            activation=activation)

        keypoints_bottleneck_channels = self.keypoints_encoder.get_num_outout_channels()[-1]
        self.keypoints_bottleneck = Sequential(
            SiarohinSameSizeBlock(
                in_channels=keypoints_bottleneck_channels + image_bottleneck_channels,
                out_channels=image_bottleneck_channels,
                activation=activation),
            ResidualBlock(
                in_channels=image_bottleneck_channels,
                one_pixel=False,
                activation=activation))

        num_skipover_channels = add_list(
            self.decomposer_encoder.get_num_outout_channels(),
            self.keypoints_encoder.get_num_outout_channels())
        num_skipover_channels[-1] = image_bottleneck_channels
        self.keypoints_decoder = SiarohinUNetDecoder(
            out_channels=decomposer_in_channels,
            num_blocks=num_blocks,
            num_skipover_channels=num_skipover_channels,
            max_channels=max_channels,
            activation=activation)

        self.alpha = Sequential(one_by_one_conv(decomposer_in_channels, 1), Sigmoid())
        self.color_change = Sequential(one_by_one_conv(decomposer_in_channels, 4), Tanh())

    def forward(self, source_image: torch.Tensor, keypoints: torch.Tensor) -> List[torch.Tensor]:
        n, c, h, w = source_image.shape

        source_image_first = self.source_image_first(source_image)
        decomposer_encoder_output = self.decomposer_encoder(source_image_first)

        if self.in_channels == 3:
            source_image = torch.cat([source_image, torch.ones(n, 1, h, w, device=source_image.device)], dim=1)

        keypoint_gaussian = keypoint_to_gaussian(keypoints, w, self.keypoint_variance)
        keypoint_encoder_output = self.keypoints_encoder(self.keypoints_first(keypoint_gaussian))
        skipover_tensors = concat_list(decomposer_encoder_output[:-1], keypoint_encoder_output[:-1])

        keypoint_bottleneck_input = torch.cat([decomposer_encoder_output[-1], keypoint_encoder_output[-1]], dim=1)
        keypoint_bottleneck_output = self.keypoints_bottleneck(keypoint_bottleneck_input)

        keypoint_decoder_input = skipover_tensors + [keypoint_bottleneck_output]
        keypoint_decoder_output = self.keypoints_decoder(keypoint_decoder_input)

        alpha = self.alpha(keypoint_decoder_output[0])
        color_change = self.color_change(keypoint_decoder_output[0])
        morphed_source_image = source_image * alpha + (1.0 - alpha) * color_change

        if self.in_channels == 3:
            morphed_source_image = morphed_source_image[:, 0:3, :, :]

        return [
            morphed_source_image,
            alpha,
            color_change
        ]



