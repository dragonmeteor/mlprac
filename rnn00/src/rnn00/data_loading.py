import glob
import os

import torch

from rnn00.util import unicode_to_ascii, string_to_one_hot_tensor, string_to_long_tensor


def add_start_and_end_letter(s: str, start_letter: str = None, end_letter: str = None):
    if start_letter is not None:
        s = start_letter + s
    if end_letter is not None:
        s = s + end_letter
    return s


def load_languages(dir: str):
    languages = []
    pattern = dir + "/*.txt"
    file_names = glob.glob(pattern)
    for file_name in file_names:
        language = os.path.splitext(os.path.basename(file_name))[0]
        languages.append(language)
    return languages


def load_languages_and_names(dir: str, start_letter: str = None, end_letter: str = None):
    languages = []
    names = []
    pattern = dir + "/*.txt"
    file_names = glob.glob(pattern)
    for file_name in file_names:
        language = os.path.splitext(os.path.basename(file_name))[0]
        languages.append(language)
        with open(file_name, "rt", encoding='utf-8') as f:
            lines = f.readlines()
            ascii_lines = [add_start_and_end_letter(unicode_to_ascii(line), start_letter, end_letter) for line in lines]
            names.append(ascii_lines)
    return languages, names


def create_examples(languages, names, letters: str, device=torch.device('cpu')):
    tensor_examples = []
    readable_examples = []
    for language_index in range(len(languages)):
        for name_index in range(len(names[language_index])):
            label = torch.tensor([language_index], dtype=torch.int64, device=device)
            one_hot_tensor = string_to_one_hot_tensor(names[language_index][name_index], letters, device)
            long_tensor = string_to_long_tensor(names[language_index][name_index], letters, device)
            tensor_examples.append((one_hot_tensor, label, long_tensor))
            readable_examples.append((names[language_index][name_index], languages[language_index]))
    return tensor_examples, readable_examples
