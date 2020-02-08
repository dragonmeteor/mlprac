import os
import string
from datetime import datetime
import time
import random

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from pytasuku import Workspace
from rnn00.data_loading import load_languages_and_names, create_examples, load_languages
from rnn00.rnn_conditional_generator import RnnConditionalGenerator
from rnn00.util import one_hot, torch_save, torch_load

DIR = "data/_20200208"

GENERATOR_FILE_NAME = DIR + "/rnn00_generator.pt"
DATA_DIR = "data/_20200206/names"

ALL_LETTERS = string.ascii_letters + " .,;'^$"
NUM_LETTERS = len(ALL_LETTERS)
NUM_LANGUAGES = 18
STATE_SIZE = 256
HIDDEN_LAYER_SIZE = 256
NUM_ITERATIONS = 1000000


def create_rnn_generator():
    return RnnConditionalGenerator(
        letters=ALL_LETTERS,
        start_letter='^',
        end_letter='$',
        num_classes=NUM_LANGUAGES,
        state_size=STATE_SIZE,
        hidden_layer_size=HIDDEN_LAYER_SIZE,
        hidden_layer_count=2,
        initialization_method='he')


def get_log_dir():
    now = datetime.now()
    return DIR + "/log/" + now.strftime("%Y_%m_%d__%H_%M_%S")


def train_generator(device: torch.device):
    print("Creating model...")
    rnn_generator = create_rnn_generator().to(device)

    print("Loading data...")
    languages, names = load_languages_and_names(DATA_DIR, start_letter='^', end_letter='$')
    examples, readable_examples = create_examples(languages, names, ALL_LETTERS, device)

    summary_writer = SummaryWriter(log_dir=get_log_dir())
    optimizer = Adam(rnn_generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    last_time = time.time()
    example_indices = []
    rnn_generator.train(True)
    loss_func = CrossEntropyLoss(reduction='mean')
    for iter_index in range(NUM_ITERATIONS):
        if len(example_indices) == 0:
            example_indices = [i for i in range(len(examples))]
            random.shuffle(example_indices)
        example_index = example_indices.pop()
        example = examples[example_index]

        name_one_hot = example[0]
        language_index = example[1].item()
        name_long = example[2]
        klass = one_hot(NUM_LANGUAGES, language_index)
        n = name_one_hot.shape[0]

        rnn_generator.zero_grad()
        scores = []
        for i in range(n - 1):
            state = rnn_generator.initial_state(device)
            letter = name_one_hot[i]
            score, state = rnn_generator.forward(letter, klass, state)
            scores.append(score)
        score_tensor = torch.cat(scores, dim=0)
        loss = loss_func(score_tensor, name_long[1:])
        loss.backward()
        optimizer.step()

        current_time = time.time()
        if current_time - last_time > 10:
            print("Processed", iter_index + 1, "iterations: loss =", loss.item())
            last_time = current_time

        if iter_index % 100 == 0:
            summary_writer.add_scalar("training_loss", loss.item(), iter_index)

    torch_save(rnn_generator.state_dict(), GENERATOR_FILE_NAME)


SAMPLES_DIR = DIR + "/samples"
SAMPLES_DONE_FILE = DIR + "/samples/done.txt"
NUM_SAMPLES_PER_LANGUAGE = 20


def generate_samples(device: torch.device = torch.device('cpu')):
    languages = load_languages(DATA_DIR)

    print("Loading generator...")
    rnn_generator = create_rnn_generator()
    rnn_generator.load_state_dict(torch_load(GENERATOR_FILE_NAME))
    rnn_generator.to(device)
    rnn_generator.train(False)

    for language_index in range(len(languages)):
        file_name = SAMPLES_DIR + "/" + languages[language_index] + ".txt"
        with open(file_name, "wt") as fout:
            for i in range(NUM_SAMPLES_PER_LANGUAGE):
                name = rnn_generator.generate(language_index, "^", device)
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                fout.write(name + "\n")

    os.makedirs(os.path.dirname(SAMPLES_DONE_FILE), exist_ok=True)
    with open(SAMPLES_DONE_FILE, "wt") as fout:
        fout.write("DONE\n")


def define_tasks(workspace: Workspace):
    device = torch.device('cpu')
    workspace.create_file_task(GENERATOR_FILE_NAME, [], lambda: train_generator(device))
    workspace.create_file_task(SAMPLES_DONE_FILE, [], lambda: generate_samples(device))


if __name__ == "__main__":
    device = torch.device('cpu')
    train_generator(device)
