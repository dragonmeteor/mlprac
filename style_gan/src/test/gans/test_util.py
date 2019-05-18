import torch


def tensor_equals(a: torch.Tensor, b: torch.Tensor, epsilon=1e-4):
    assert a.shape == b.shape
    return a.sub(b).abs().max() < epsilon