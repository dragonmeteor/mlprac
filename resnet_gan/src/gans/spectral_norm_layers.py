import torch

from torch.nn import Module, Parameter
from torch.nn.functional import conv2d, linear
from torch.nn.init import kaiming_normal_


# This implementation is based on https://github.com/balansky/pytorch_gan/blob/master/nets/layers/spectral_norm.py
def normalize(v: torch.Tensor):
    return v / (v.norm() + 1e-12)


def max_singular_value(M, u: torch.Tensor):
    with torch.no_grad():
        _v = normalize(torch.matmul(u, M))
        _u = normalize(torch.matmul(_v, torch.t(M)))
    sigma = torch.sum((torch.matmul(_u, M) * _v))
    return _u, sigma, _v


class SnConv2d(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        kaiming_normal_(self.weight)
        self.bias = Parameter(torch.zeros(out_channels))
        self.register_buffer("u", torch.randn(1, out_channels, requires_grad=False))

    def forward(self, input):
        W = self.weight.view(self.out_channels, -1)
        u, sigma, _ = max_singular_value(W, self.u.clone().detach())
        if sigma > 1:
            W_bar = self.weight / sigma
        else:
            W_bar = self.weight
        if self.training:
            self.u.copy_(u)
        return conv2d(input=input, weight=W_bar, bias=self.bias, stride=self.stride, padding=self.padding)


class SnLinear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features))
        kaiming_normal_(self.weight)
        self.bias = Parameter(torch.zeros(out_features))
        self.register_buffer("u", torch.randn(1, out_features, requires_grad=False))

    def forward(self, input):
        u, sigma, _ = max_singular_value(self.weight, self.u.clone().detach())
        if sigma > 1:
            W_bar = self.weight / sigma
        else:
            W_bar = self.weight
        if self.training:
            self.u.copy_(u)
        return linear(input, W_bar, self.bias)


if __name__ == "__main__":
    #cuda = torch.device('cuda')
    #A = SnLinear(in_features=3, out_features=3).to(cuda)
    #x = torch.ones(1, 3, device=cuda)
    #for i in range(100):
    #    A(x)
    #    print(A.u.norm())
    #    print(max_singular_value(A.weight, A.u))

    print(torch.__version__)
