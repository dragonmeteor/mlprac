import torch
import unittest

from gans.simplified_style_gan import AdaIN
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
        y = AdaIN(x, torch.Tensor([-1,0,1]), torch.Tensor([10,100,100]))

        self.assertTrue(y.shape == torch.Size((1,3,3,3)))

        y_flattened = y.view(1,3,9)
        y_mean = y_flattened.mean(dim=2, keepdim=True).view(3)
        y_std = y_flattened.std(dim=2, keepdim=True).view(3)

        self.assertTrue(tensor_equals(y_mean, torch.Tensor([-1, 0, 1])))
        self.assertTrue(tensor_equals(y_std, torch.Tensor([10,100,100])))


class GeneratorBlockTests(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
