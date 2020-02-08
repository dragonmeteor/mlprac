import torch
from torch.distributions import Categorical
from torch.nn import Module, Sequential, Linear, Softmax

from rnn00.init_func import create_init_function
from rnn00.linear import LinearBlock
from rnn00.util import one_hot


class RnnConditionalGenerator(Module):
    def __init__(self,
                 letters: str,
                 start_letter: str,
                 end_letter: str,
                 num_classes: int,
                 state_size: int,
                 hidden_layer_size: int = 256,
                 hidden_layer_count: int = 2,
                 initialization_method='he'):
        super().__init__()
        self.letters = letters
        self.start_letter = start_letter
        self.start_letter_index = self.letters.find(start_letter)
        assert 0 <= self.start_letter_index < len(self.letters)
        self.end_letter = end_letter
        self.end_letter_index = self.letters.find(end_letter)
        assert 0 <= self.end_letter_index < len(self.letters)
        self.num_letters = len(letters)
        self.num_classes = num_classes
        self.state_size = state_size
        assert hidden_layer_size > 0
        assert hidden_layer_count > 1
        input_size = self.num_letters + self.num_classes + self.state_size
        init = create_init_function(initialization_method)

        input_to_state_layers = [LinearBlock(input_size, hidden_layer_size, initialization_method)]
        for i in range(hidden_layer_count - 1):
            input_to_state_layers.append(LinearBlock(hidden_layer_size, hidden_layer_size, initialization_method))
        input_to_state_layers.append(LinearBlock(hidden_layer_size, state_size, initialization_method))
        self.input_to_state = Sequential(*input_to_state_layers)
        self.state_to_score = init(Linear(state_size, self.num_letters, bias=True))

    def forward(self, letter: torch.Tensor, klass: torch.Tensor, state: torch.Tensor):
        combined = torch.cat([letter, klass, state], dim=1)
        state = self.input_to_state(combined)
        score = self.state_to_score(state)
        return score, state

    def forward_and_sample(self, letter: torch.Tensor, klass: torch.Tensor, state: torch.Tensor):
        score, state = self.forward(letter, klass, state)
        distribution = Categorical(torch.softmax(score, dim=1).squeeze(0))
        next_letter_index = distribution.sample((1,)).item()
        next_letter = one_hot(self.num_letters, next_letter_index, letter.device)
        return score, state, next_letter, self.letters[next_letter_index]

    def initial_state(self, device: torch.device = torch.device('cpu')):
        return torch.zeros(1, self.state_size, device=device)

    def generate(self, class_index: int, initial_letter: str = None, device: torch.device = torch.device('cpu')):
        if initial_letter == None:
            initial_letter = self.start_letter
        letter_index = self.letters.find(initial_letter)
        assert 0 <= letter_index < self.num_letters
        state = self.initial_state(device)
        if letter_index == self.end_letter_index:
            return "", state

        letter = one_hot(self.num_letters, letter_index, device)
        klass = one_hot(self.num_classes, class_index, device)
        output_str = ''
        while letter_index != self.end_letter_index:
            score, next_state, next_letter, next_letter_str = self.forward_and_sample(letter, klass, state)
            output_str += next_letter_str
            state = next_state
            letter = next_letter
            letter_index = self.letters.find(letter)
        return output_str, state


if __name__ == "__main__":
    print(torch.tensor([[0.5, 0.25, 0.25]]).squeeze(0))
    probs = torch.tensor([0.5, 0.25, 0.25])
    print(probs.shape)
    categorical = Categorical(probs)
    print(categorical.sample((10,)))
