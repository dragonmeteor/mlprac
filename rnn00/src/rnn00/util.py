import os
import string

import torch
import unicodedata

ASCII_LETTERS_PLUS_ALPHA = string.ascii_letters + " .,;'"


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in ASCII_LETTERS_PLUS_ALPHA)


def letter_to_index(letter: str, letters: str):
    return letters.find(letter)


def string_to_one_hot_tensor(s: str, letters: str, device: torch.device = torch.device('cpu')):
    tensor = torch.zeros(len(s), 1, len(letters), device=device)
    for i in range(len(s)):
        tensor[i][0][letter_to_index(s[i], letters)] = 1.0
    return tensor


def string_to_long_tensor(s: str, letters: str, device: torch.device = torch.device('cpu')):
    data = [letter_to_index(s[i], letters) for i in range(len(s))]
    return torch.tensor(data, dtype=torch.int64, device=device)


def torch_save(content, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        torch.save(content, f)


def torch_load(file_name):
    with open(file_name, 'rb') as f:
        return torch.load(f)


def one_hot(num_classes: int, index: int, device: torch.device = torch.device('cpu')):
    output = torch.zeros(1, num_classes, device=device)
    output[0, index] = 1
    return output


if __name__ == "__main__":
    print(ALL_LETTERS)
    john = string_to_one_hot_tensor("John")
    print(john.shape)
    print(john)
