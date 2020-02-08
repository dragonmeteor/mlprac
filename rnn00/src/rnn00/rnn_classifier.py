import torch
from torch.nn import Module, Linear, Sequential

from rnn00.init_func import create_init_function
from rnn00.linear import LinearBlock


class RnnClassifier(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, initialization_method='he'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        init = create_init_function(initialization_method)

        self.i2h = Sequential(
            LinearBlock(input_size + hidden_size, 256, initialization_method),
            LinearBlock(256, 256, initialization_method),
            LinearBlock(256, 256, initialization_method),
            init(Linear(256, hidden_size)))
        self.i2o = Sequential(
            LinearBlock(input_size + hidden_size, 256, initialization_method),
            LinearBlock(256, 256, initialization_method),
            LinearBlock(256, 256, initialization_method),
            init(Linear(256, output_size)))

    def initial_hidden_state(self, device=torch.device('cpu')):
        return torch.zeros(1, self.hidden_size, device=device)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        combined = torch.cat([input, hidden], dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def classify(self, input: torch.Tensor, device=torch.device('cpu')):
        hidden = self.initial_hidden_state(device)
        n = input.shape[0]
        assert n > 0
        output = None
        for i in range(n):
            output, hidden = self.forward(input[i], hidden)
        return output
