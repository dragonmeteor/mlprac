import torch
import unittest

from gans.simplified_style_gan import AdaIN, GeneratorBlock, create_noise, GeneratorFirstBlock, GeneratorModule, \
    GeneratorTransitionModule
from test.gans.test_util import tensor_equals


class AdaINTest(unittest.TestCase):
    def test_adaIn(self):
        x = torch.Tensor([[
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]
            ],
            [
                [5, 10, 15],
                [5, 10, 15],
                [5, 10, 15],
            ],
            [
                [98, 100, 102],
                [98, 100, 102],
                [98, 100, 102],
            ]
        ]])
        y = AdaIN(x, torch.Tensor([-1, 0, 1]), torch.Tensor([10, 100, 100]))

        self.assertEquals(y.shape, torch.Size((1, 3, 3, 3)))

        y_flattened = y.view(1, 3, 9)
        y_mean = y_flattened.mean(dim=2, keepdim=True).view(3)
        y_std = y_flattened.std(dim=2, keepdim=True).view(3)

        self.assertTrue(tensor_equals(y_mean, torch.Tensor([-1, 0, 1])))
        self.assertTrue(tensor_equals(y_std, torch.Tensor([10, 100, 100])))


class CreateNoiseTests(unittest.TestCase):
    def test_create_noise__noise_input_is_none(self):
        h = 256
        w = 256

        block = GeneratorBlock(in_channels=4, out_channels=4)
        input_image = torch.ones(3, 4, h, w)

        noise = create_noise(input_image, torch.Tensor([1, 2, 3, 4]).view(1, 4, 1, 1), None)

        self.assertEquals(block.noise_1_factor.shape, torch.Size((1, 4, 1, 1)))
        self.assertEquals(block.noise_2_factor.shape, torch.Size((1, 4, 1, 1)))
        self.assertEquals(noise.shape, torch.Size((3, 4, h, w)))

        std_diff = noise.view(3, 4, h * w).std(dim=2) - torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
        self.assertLess(std_diff.abs().max(), 0.1)

        self.assertLess(noise.view(3, 4, h * w).mean(dim=2).abs().max(), 0.1)

    def test_create_noise__noise_input_is_not_none(self):
        h = 4
        w = 4

        block = GeneratorBlock(in_channels=4, out_channels=4)
        input_image = torch.ones(3, 4, h, w)

        noise = create_noise(
            input_image,
            noise_factor=torch.Tensor([1, 2, 3, 4]).view(1, 4, 1, 1),
            input_noise=torch.Tensor([
                [0.01, 0.02, 0.03, 0.04],
                [0.1, 0.2, 0.3, 0.4],
                [1, 2, 3, 4],
                [10, 20, 30, 40]
            ])
        )

        self.assertEquals(noise.shape, torch.Size((3, 4, h, w)))
        self.assertTrue(tensor_equals(noise[0], torch.Tensor([
            [
                [0.01, 0.02, 0.03, 0.04],
                [0.1, 0.2, 0.3, 0.4],
                [1, 2, 3, 4],
                [10, 20, 30, 40]
            ],
            [
                [0.02, 0.04, 0.06, 0.08],
                [0.2, 0.4, 0.6, 0.8],
                [2, 4, 6, 8],
                [20, 40, 60, 80]
            ],
            [
                [0.03, 0.06, 0.09, 0.12],
                [0.3, 0.6, 0.9, 1.2],
                [3, 6, 9, 12],
                [30, 60, 90, 120]
            ],
            [
                [0.04, 0.08, 0.12, 0.16],
                [0.4, 0.8, 1.2, 1.6],
                [4, 8, 12, 16],
                [40, 80, 120, 160]
            ],
        ])))
        self.assertTrue(tensor_equals(noise[0], noise[1]))
        self.assertTrue(tensor_equals(noise[0], noise[2]))


class GeneratorBlockTests(unittest.TestCase):
    def test_upsample(self):
        block = GeneratorBlock(in_channels=16, out_channels=16)
        input_image = torch.ones(3, 16, 4, 4)

        upsampled_image = block.upsample(input_image)

        self.assertEquals(upsampled_image.shape, torch.Size((3, 16, 8, 8)))
        self.assertEquals((upsampled_image - 1).abs().max(), 0.0)

    def test_convolve_1(self):
        block = GeneratorBlock(in_channels=16, out_channels=32)
        input_image = torch.ones(3, 16, 8, 8)

        conv_1_image = block.convolve_1(input_image)

        self.assertEquals(conv_1_image.shape, torch.Size((3, 32, 8, 8)))

    def test_forward(self):
        h = 16
        w = 16

        block = GeneratorBlock(in_channels=32, out_channels=64)
        input_image = torch.ones(3, 32, h, w)
        weight = torch.ones(3, 512)

        output_image = block.forward(input_image, weight)

        self.assertEqual(output_image.shape, torch.Size((3, 64, 32, 32)))


class GeneratorFirstBlockTests(unittest.TestCase):
    def test_forward(self):
        block = GeneratorFirstBlock(image_size=4, out_channels=64)
        weight = torch.ones(3, 512)

        output_image = block.forward(weight)

        self.assertEqual(output_image.shape, torch.Size((3, 64, 4, 4)))


class GeneratorNetworkTests(unittest.TestCase):
    def test_forward(self):
        cuda = torch.device('cuda')
        network = GeneratorModule(64).to(cuda)
        latent_vector = torch.zeros(3, 512, device=cuda)

        output = network(latent_vector)

        self.assertEqual(output.shape, torch.Size((3, 3, 64, 64)))


class GeneratorTransitionNetworkTests(unittest.TestCase):
    def test_forward(self):
        cuda = torch.device('cuda')
        network = GeneratorTransitionModule(64).to(cuda)
        latent_vector = torch.zeros(3, 512, device=cuda)

        output = network(latent_vector)

        self.assertEqual(output.shape, torch.Size((3, 3, 64, 64)))


if __name__ == "__main__":
    unittest.main()
