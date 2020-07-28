from typing import List

import torch
from torch.nn import Module, Sequential, Conv2d, Sigmoid, Tanh
from torch.nn.functional import affine_grid, grid_sample

from hana.bougain.nn.common import keypoint_to_gaussian
from hana.bougain.nn.residual_block import ResidualBlock
from hana.bougain.nn.self_attention import SelfAttention
from hana.bougain.nn.siarohin_modules import SiarohinUNetEncoder, SiarohinSameSizeBlock, SiarohinUNetDecoder
from hana.rindou.nn2.init_function import create_init_function


def one_by_one_conv(in_channels: int, out_channels: int):
    init = create_init_function('he')
    return init(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))


def add_list(a: List[int], b: List[int]) -> List[int]:
    output = []
    for i in range(len(a)):
        output.append(a[i] + b[i])
    return output


def concat_list(a: List[torch.Tensor], b: List[torch.Tensor]) -> List[torch.Tensor]:
    output = []
    for i in range(len(a)):
        output.append(torch.cat([a[i], b[i]], dim=1))
    return output


class FaceDecomposerAndMorpher(Module):
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
        self.decomposer_bottleneck = Sequential(
            SelfAttention(
                in_channels=image_bottleneck_channels,
                hidden_channels=max(64, image_bottleneck_channels // 8)),
            ResidualBlock(
                in_channels=image_bottleneck_channels,
                one_pixel=False,
                activation=activation))

        self.decomposer_decoder = SiarohinUNetDecoder(
            out_channels=decomposer_in_channels,
            num_blocks=num_blocks,
            num_skipover_channels=self.decomposer_encoder.get_num_outout_channels(),
            max_channels=max_channels,
            activation=activation)

        self.iris_layer_alpha = Sequential(one_by_one_conv(decomposer_in_channels, 4), Sigmoid())
        self.iris_layer_color_change = Sequential(one_by_one_conv(decomposer_in_channels, 4), Tanh())

        self.face_layer_alpha = Sequential(one_by_one_conv(decomposer_in_channels, 4), Sigmoid())
        self.face_layer_color_change = Sequential(one_by_one_conv(decomposer_in_channels, 4), Tanh())

        self.top_layer_alpha = Sequential(one_by_one_conv(decomposer_in_channels, 4), Sigmoid())
        self.top_layer_color_change = Sequential(one_by_one_conv(decomposer_in_channels, 4), Tanh())

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
            self.decomposer_decoder.get_num_output_channels(),
            self.keypoints_encoder.get_num_outout_channels())
        num_skipover_channels[-1] = image_bottleneck_channels
        self.keypoints_decoder = SiarohinUNetDecoder(
            out_channels=decomposer_in_channels,
            num_blocks=num_blocks,
            num_skipover_channels=num_skipover_channels,
            max_channels=max_channels,
            activation=activation)

        self.iris_layer_grid_change = one_by_one_conv(decomposer_in_channels, 2)

        self.face_layer_mod_alpha = Sequential(one_by_one_conv(decomposer_in_channels, 4), Sigmoid())
        self.face_layer_mod_color_change = Sequential(one_by_one_conv(decomposer_in_channels, 4), Tanh())

    def forward(self, source_image: torch.Tensor, keypoints: torch.Tensor) -> List[torch.Tensor]:
        n, c, h, w = source_image.shape

        source_image_first = self.source_image_first(source_image)
        decomposer_encoder_output = self.decomposer_encoder(source_image_first)
        decomposer_bottleneck_output = decomposer_encoder_output[:-1] + \
                                       [self.decomposer_bottleneck(decomposer_encoder_output[-1])]
        decomposer_decoder_output = self.decomposer_decoder(decomposer_bottleneck_output)

        if self.in_channels == 3:
            source_image = torch.cat([source_image, torch.ones(n, 1, h, w, device=source_image.device)], dim=1)

        background_layer = source_image
        iris_layer = self.compute_layer(
            self.iris_layer_alpha, self.iris_layer_color_change, source_image, decomposer_decoder_output[0])
        face_layer = self.compute_layer(
            self.face_layer_alpha, self.face_layer_color_change, source_image, decomposer_decoder_output[0])
        top_layer = self.compute_layer(
            self.top_layer_alpha, self.top_layer_color_change, source_image, decomposer_decoder_output[0])

        reconstructed_source_image = self.merge_down_layers([background_layer, iris_layer, face_layer, top_layer])
        if self.in_channels == 3:
            reconstructed_source_image = reconstructed_source_image[:, 0:3, :, :]

        keypoint_gaussian = keypoint_to_gaussian(keypoints, w, self.keypoint_variance)
        keypoint_encoder_output = self.keypoints_encoder(self.keypoints_first(keypoint_gaussian))
        skipover_tensors = concat_list(decomposer_decoder_output[:-1], keypoint_encoder_output[:-1])

        keypoint_bottleneck_input = torch.cat([keypoint_encoder_output[-1], decomposer_decoder_output[-1]], dim=1)
        keypoint_bottleneck_output = self.keypoints_bottleneck(keypoint_bottleneck_input)

        keypoint_decoder_input = skipover_tensors + [keypoint_bottleneck_output]
        keypoint_decoder_output = self.keypoints_decoder(keypoint_decoder_input)

        iris_layer_grid_change = self.iris_layer_grid_change(keypoint_decoder_output[0]).permute((0, 2, 3, 1))
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) \
            .to(source_image.device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=False)
        iris_layer_grid = base_grid + iris_layer_grid_change
        modded_iris_layer = grid_sample(iris_layer, iris_layer_grid, mode='bilinear', padding_mode='border',
                                        align_corners=False)

        modded_face_layer = self.compute_layer(self.face_layer_mod_alpha, self.face_layer_mod_color_change,
                                               face_layer, keypoint_decoder_output[0])

        morphed_source_image = self.merge_down_layers(
            [background_layer, modded_iris_layer, modded_face_layer, top_layer])
        if self.in_channels == 3:
            morphed_source_image = morphed_source_image[:, 0:3, :, :]

        return [
            reconstructed_source_image,
            morphed_source_image,
            iris_layer,
            face_layer,
            top_layer,
            modded_iris_layer,
            modded_face_layer,
        ]

    def compute_layer(self,
                      alpha_module: Module,
                      color_change_module: Module,
                      source_image: torch.Tensor,
                      features: torch.Tensor) -> torch.Tensor:
        alpha = alpha_module(features)
        color_change = color_change_module(features)
        return source_image * alpha + color_change * (1.0 - alpha)

    def merge_down_layer(self, bottom_layer, top_layer):
        bottom_layer_rgb = bottom_layer[:, 0:3, :, :]
        bottom_layer_alpha = bottom_layer[:, 3:4, :, :]

        top_layer_rgb = top_layer[:, 0:3, :, :]
        top_layer_alpha = top_layer[:, 3:4, :, :]

        merged_rgb = (1.0 - top_layer_alpha) * bottom_layer_rgb + top_layer_alpha * top_layer_rgb
        merged_alpha = (1.0 - top_layer_alpha) * bottom_layer_alpha + top_layer_alpha

        return torch.cat([merged_rgb, merged_alpha], dim=1)

    def merge_down_layers(self, layers):
        layer = layers[0]
        for i in range(1, len(layers)):
            layer = self.merge_down_layer(layer, layers[i])
        return layer


if __name__ == "__main__":
    cuda = torch.device("cuda")
    morpher = FaceDecomposerAndMorpher().to(cuda)
    source_image = torch.zeros(16, 3, 256, 256, device=cuda)
    keypoints = torch.zeros(16, 32, 2, device=cuda)
    output = morpher(source_image, keypoints)
