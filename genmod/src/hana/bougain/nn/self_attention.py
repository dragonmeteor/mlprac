import torch
from torch.nn import Module, Parameter, Conv2d


# Code inspired from from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
from hana.rindou.nn2.init_function import create_init_function


class SelfAttention(Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        init = create_init_function('he')
        self.hidden_channels = hidden_channels
        self.W_f = init(Conv2d(
            in_channels=in_channels, out_channels=hidden_channels,
            kernel_size=1, stride=1, padding=0, bias=False))
        self.W_g = init(Conv2d(
            in_channels=in_channels, out_channels=hidden_channels,
            kernel_size=1, stride=1, padding=0, bias=False))
        self.W_v = init(Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=1, stride=1, padding=0, bias=False))
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        [n, c, h, w] = x.shape

        f = torch.transpose(self.W_f(x).view(n,self.hidden_channels,h*w), dim0=1, dim1=2)
        g = self.W_g(x).view(n,self.hidden_channels,h*w)
        s = torch.bmm(f,g)
        beta = torch.softmax(s, dim=1)

        v = self.W_v(x).view(n,c,h*w)
        o = torch.bmm(v, beta).view(n, c, h, w)
        return x + self.gamma * o


if __name__ == "__main__":
    cuda = torch.device("cuda")
    x = torch.zeros(16, 1024, 8, 8, device=cuda)
    self_attention = SelfAttention(in_channels=1024, hidden_channels=1024//8).to(cuda)
    y = self_attention(x)
    print(y.shape)




