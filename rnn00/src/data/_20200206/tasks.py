import random
import string
import time

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from pytasuku import Workspace
from rnn00.data_loading import load_languages_and_names, create_examples
from rnn00.rnn_classifier import RnnClassifier
from rnn00.util import torch_save, torch_load

DIR = "data/_20200206"

ALL_LETTERS = string.ascii_letters + " .,;'"
NUM_LETTERS = len(ALL_LETTERS)
NUM_LANGUAGES = 18
HIDDEN_SIZE = 128
NUM_ITERATIONS = 100000

RNN_00_FILE = DIR + "/rnn00.pt"


def train_rnn00(device=torch.device('cpu')):
    print("Creating the model...")
    rnn00 = RnnClassifier(NUM_LETTERS, HIDDEN_SIZE, NUM_LANGUAGES).to(device)
    rnn00.train(True)

    print("Loading training examples...")
    languages, names = load_languages_and_names("data/_20200206/names")
    examples, readable_examples = create_examples(languages, names, ALL_LETTERS, device)

    summary_writer = SummaryWriter(log_dir=DIR + "/log")

    optimizer = Adam(rnn00.parameters(), lr=0.0001, betas=(0.5, 0.999))

    loss = CrossEntropyLoss()
    indices = []
    last_time = time.time()
    for iter_index in range(NUM_ITERATIONS):
        if len(indices) == 0:
            indices = [i for i in range(len(examples))]
            random.shuffle(indices)
        index = indices.pop()
        name, label = examples[index]

        rnn00.zero_grad()
        output = rnn00.classify(name, device)
        loss_value = loss(output, label)
        loss_value.backward()
        optimizer.step()

        current_time = time.time()
        if current_time - last_time > 10:
            print("Done", iter_index, "iterations ...")
            readable_example = readable_examples[index]
            top_values, top_indices = output.topk(1)
            print("Name =", readable_example[0],
                  ", correct =", readable_example[1],
                  ", output =", languages[top_indices[0].item()])
            last_time = current_time

        if iter_index % 100 == 0:
            summary_writer.add_scalar("training_loss", loss_value.item(), iter_index)

    torch_save(rnn00.state_dict(), RNN_00_FILE)


def evaluate_rnn00(device=torch.device('cpu')):
    rnn00 = RnnClassifier(NUM_LETTERS, HIDDEN_SIZE, NUM_LANGUAGES).to(device)
    rnn00.load_state_dict(torch_load(RNN_00_FILE))
    rnn00.to(device)
    rnn00.train(False)

    languages, names = load_languages_and_names("data/_20200206/names")
    examples, readable_examples = create_examples(languages, names, ALL_LETTERS, device)

    correct = 0
    for index in range(len(examples)):
        name, label = examples[index]
        output = rnn00.classify(name, device)
        _, top_indices = output.topk(1)
        readable_example = readable_examples[index]
        top_values, top_indices = output.topk(1)
        if top_indices[0].item() == label.item():
            correct += 1
        print("Name =", readable_example[0],
              ", correct =", readable_example[1],
              ", output =", languages[top_indices[0].item()])

    print("accuracy = %f%%" % (correct * 100.0 / len(examples)))


def define_tasks(workspace: Workspace):
    device = torch.device('cpu')
    workspace.create_file_task(RNN_00_FILE, [], lambda: train_rnn00(device))
    workspace.create_command_task(DIR + "/evaluate", [RNN_00_FILE], lambda: evaluate_rnn00(device))


if __name__ == "__main__":
    # languages, names = load_languages_and_names()
    # examples = create_examples(languages, names)
    # print(len(examples))
    train_rnn00(torch.device('cuda'))
