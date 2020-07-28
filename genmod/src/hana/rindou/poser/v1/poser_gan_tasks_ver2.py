import abc
import os
import random
import shutil
import time
from datetime import datetime
from typing import List, Tuple

import PIL.Image
import matplotlib.pyplot as plt
import numpy
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec
from hana.rindou.util import torch_save, save_rng_state, torch_load, load_rng_state, optimizer_to_device, \
    rgba_to_numpy_image_greenscreen
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from pytasuku.indexed.one_index_file_tasks import OneIndexFileTasks


class PoserGanTrainingSpecVer2:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def save_point_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def example_per_save_point(self) -> int:
        pass

    @abc.abstractmethod
    def generator_learning_rate(self, save_point_index: int, global_example_count: int) -> float:
        pass

    @abc.abstractmethod
    def discriminator_learning_rate(self, save_point_index: int, global_example_count: int) -> float:
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def generator_betas(self) -> Tuple[float, float]:
        pass

    @property
    @abc.abstractmethod
    def discriminator_betas(self) -> Tuple[float, float]:
        pass

    @property
    def iter_per_save_point(self) -> int:
        output = self.example_per_save_point // self.batch_size
        if self.example_per_save_point % self.batch_size != 0:
            output += 1
        return output

    @property
    def example_seen_per_save_point(self) -> int:
        return self.batch_size * self.iter_per_save_point

    @property
    @abc.abstractmethod
    def random_seed(self) -> int:
        pass


class PoserGanValidationSpecVer2:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def example_per_batch(self) -> int:
        pass


class PoserGanSampleOutputSpecVer2:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def example_per_row(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def example_per_sample_output(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def sample_output_index_seed(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def image_size(self) -> int:
        pass

    def save_sample_image(self, G: Module, sample_output_batch, file_name: str):
        G.train(False)
        source_image = sample_output_batch[0]
        pose = sample_output_batch[1]
        target_image = sample_output_batch[2]

        output = G(source_image, pose)
        self.save_output_image(source_image, target_image, output[0].detach(), file_name)

    def save_output_image(self, source_images, target_images, output_images, file_name):
        source_images = F.interpolate(source_images, size=self.image_size).cpu()
        target_images = F.interpolate(target_images, size=self.image_size).cpu()
        output_images = F.interpolate(output_images, size=self.image_size).cpu()

        n = output_images.shape[0]
        num_rows = n // self.example_per_row
        if n % self.example_per_row != 0:
            num_rows += 1
        num_cols = 3 * self.example_per_row

        image_size = self.image_size
        output = numpy.zeros([num_rows * image_size, num_cols * image_size, 3])
        for i in range(n):
            row = i // self.example_per_row
            col = i % self.example_per_row

            source_image = rgba_to_numpy_image_greenscreen(source_images[i])
            target_image = rgba_to_numpy_image_greenscreen(target_images[i])
            output_image = rgba_to_numpy_image_greenscreen(output_images[i])

            row_start = row * image_size
            row_end = row_start + image_size

            col_start = (3 * col) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = source_image

            col_start = (3 * col + 1) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = target_image

            col_start = (3 * col + 2) * image_size
            col_end = col_start + image_size
            output[row_start:row_end, col_start:col_end, :] = output_image

        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(output * 255.0)), mode='RGB')
        pil_image.save(file_name)
        print("Saved %s" % file_name)

    def plot_image(self, image, num_rows, num_cols, row, col):
        index = row * num_cols + col + 1
        ax = plt.subplot(num_rows, num_cols, index)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image)


def set_learning_rate(module, lr):
    for param_group in module.param_groups:
        param_group['lr'] = lr


class PoserGanTasksVer2:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 gan_spec: PoserGanSpec,
                 loss_spec: PoserGanLoss,
                 training_dataset: Dataset,
                 validation_dataset: Dataset,
                 training_spec: PoserGanTrainingSpecVer2,
                 validation_spec: PoserGanValidationSpecVer2,
                 sample_output_spec: PoserGanSampleOutputSpecVer2,
                 pretrained_generator_file_name: str = None,
                 pretrained_discriminator_file_name: str = None,
                 device=torch.device('cpu')):
        self.workspace = workspace
        self.prefix = prefix
        self.gan_spec = gan_spec
        self.loss_spec = loss_spec
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.training_spec = training_spec
        self.validation_spec = validation_spec
        self.sample_output_spec = sample_output_spec
        self.pretrained_generator_file_name = pretrained_generator_file_name
        self.pretrained_discriminator_file_name = pretrained_discriminator_file_name
        self.device = device

        # Initial models
        self.sample_output_indices_tasks = SampleOutputIndicesTasks(self)
        self.initial_generator_tasks = InitialGeneratorTasks(self)
        self.initial_discriminator_tasks = InitialDiscriminatorTasks(self)
        self.initial_rng_state_tasks = InitialRngStateTasks(self)

        # Training tasks
        self.rng_state_tasks = RngStateTasks(self)
        self.generator_tasks = GeneratorTasks(self)
        self.discriminator_tasks = DiscriminatorTasks(self)
        self.generator_optimizer_tasks = GeneratorOptimizerTasks(self)
        self.discriminator_optimizer_tasks = DiscriminatorOptimizerTasks(self)

        self.rng_state_tasks.define_tasks()
        self.generator_tasks.define_tasks()
        self.discriminator_tasks.define_tasks()
        self.generator_optimizer_tasks.define_tasks()
        self.discriminator_optimizer_tasks.define_tasks()

        self.training_data_loader = None
        self.training_data_loader_iter = None
        self.validation_data_loader = None
        self.validation_data_loader_iter = None
        self.sample_output_batch = None
        self.summary_writer = None
        self.log_dir = None

    def get_log_dir(self):
        if self.log_dir is None:
            now = datetime.now()
            self.log_dir = self.prefix + "/log/" + now.strftime("%Y_%m_%d__%H_%M_%S")
        return self.log_dir

    def get_summary_writer(self):
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(log_dir=self.get_log_dir())
        return self.summary_writer

    def save_initial_models(self):
        torch.manual_seed(self.training_spec.random_seed)

        if self.pretrained_generator_file_name is None:
            generator = self.gan_spec.generator().to(self.device)
            torch_save(generator.state_dict(), self.initial_generator_tasks.file_name)
        else:
            shutil.copyfile(self.pretrained_generator_file_name, self.initial_generator_tasks.file_name)

        if self.pretrained_discriminator_file_name is None:
            discriminator = self.gan_spec.discriminator().to(self.device)
            torch_save(discriminator.state_dict(), self.initial_discriminator_tasks.file_name)
        else:
            shutil.copyfile(self.pretrained_discriminator_file_name, self.initial_discriminator_tasks.file_name)

        save_rng_state(self.initial_rng_state_tasks.file_name)

    def save_point_dependencies(self, save_point_index: int) -> List[str]:
        if save_point_index == 0:
            return [
                self.initial_rng_state_tasks.file_name,
                self.initial_generator_tasks.file_name,
                self.initial_discriminator_tasks.file_name,
                self.sample_output_indices_tasks.file_name,
            ]
        else:
            return [
                self.sample_output_indices_tasks.file_name,
                self.rng_state_tasks.file_name(save_point_index - 1),
                self.generator_tasks.file_name(save_point_index - 1),
                self.discriminator_tasks.file_name(save_point_index - 1),
                self.generator_optimizer_tasks.file_name(save_point_index - 1),
                self.discriminator_optimizer_tasks.file_name(save_point_index - 1)
            ]

    def process_save_point(self, save_point_index: int):
        print("in process save point")
        if save_point_index == 0:
            self.save_save_point_zero_files()
        else:
            self.train(save_point_index)

    def save_save_point_zero_files(self):
        load_rng_state(self.initial_rng_state_tasks.file_name)
        G = self.load_generator(self.initial_generator_tasks.file_name)
        D = self.load_discriminator(self.initial_discriminator_tasks.file_name)
        G_optim = Adam(G.parameters(),
                       lr=self.training_spec.generator_learning_rate(0, 0),
                       betas=self.training_spec.generator_betas)
        D_optim = Adam(D.parameters(),
                       lr=self.training_spec.discriminator_learning_rate(0, 0),
                       betas=self.training_spec.discriminator_betas)

        torch_save(G.state_dict(), self.generator_tasks.file_name(0))
        torch_save(D.state_dict(), self.discriminator_tasks.file_name(0))
        torch_save(G_optim.state_dict(), self.generator_optimizer_tasks.file_name(0))
        torch_save(D_optim.state_dict(), self.discriminator_optimizer_tasks.file_name(0))
        save_rng_state(self.rng_state_tasks.file_name(0))

    def load_generator(self, file_name):
        G = self.gan_spec.generator().to(self.device)
        G.load_state_dict(torch_load(file_name))
        return G

    def load_discriminator(self, file_name):
        D = self.gan_spec.discriminator().to(self.device)
        D.load_state_dict(torch_load(file_name))
        return D

    def load_generator_optimizer(self, G: Module, save_point: int):
        G_optim = Adam(G.parameters(),
                       lr=self.training_spec.generator_learning_rate(save_point, 0),
                       betas=self.training_spec.generator_betas)
        file_name = self.generator_optimizer_tasks.file_name(save_point)
        G_optim.load_state_dict(torch_load(file_name))
        optimizer_to_device(G_optim, self.device)
        return G_optim

    def load_discriminator_optimizer(self, D: Module, save_point: int):
        D_optim = Adam(D.parameters(),
                       lr=self.training_spec.discriminator_learning_rate(save_point, 0),
                       betas=self.training_spec.discriminator_betas)
        file_name = self.discriminator_optimizer_tasks.file_name(save_point)
        D_optim.load_state_dict(torch_load(file_name))
        optimizer_to_device(D_optim, self.device)
        return D_optim

    def get_next_training_batch(self):
        if self.training_data_loader is None:
            self.training_data_loader = DataLoader(
                self.training_dataset,
                batch_size=self.training_spec.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True)
        if self.training_data_loader_iter is None:
            self.training_data_loader_iter = iter(self.training_data_loader)
        try:
            batch = next(self.training_data_loader_iter)
        except StopIteration:
            self.training_data_loader_iter = iter(self.training_data_loader)
            batch = next(self.training_data_loader_iter)
        return [x.to(self.device) for x in batch]

    def get_next_validation_batch(self):
        if self.validation_data_loader is None:
            self.validation_data_loader = DataLoader(
                self.validation_dataset,
                batch_size=self.validation_spec.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True)
        if self.validation_data_loader_iter is None:
            self.validation_data_loader_iter = iter(self.validation_data_loader)
        try:
            batch = next(self.validation_data_loader_iter)
        except StopIteration:
            self.validation_data_loader_iter = iter(self.training_data_loader)
            batch = next(self.validation_data_loader_iter)
        return [x.to(self.device) for x in batch]

    def get_sample_output_batch(self):
        if self.sample_output_batch is not None:
            return self.sample_output_batch

        examples = []
        sample_output_indices = torch_load(self.sample_output_indices_tasks.file_name)
        for index in sample_output_indices:
            example = self.validation_dataset[index]
            example = [x.to(self.device).unsqueeze(0) for x in example]
            examples.append(example)
        k = len(examples[0])
        transposed = [[] for i in range(k)]
        for example in examples:
            for i in range(k):
                transposed[i].append(example[i])
        self.sample_output_batch = [torch.cat(x, dim=0) for x in transposed]

        return self.sample_output_batch

    def save_sample_output_indices(self):
        n = len(self.validation_dataset)
        random.seed(self.sample_output_spec.sample_output_index_seed)
        sample_output_indices = random.sample(range(n), self.sample_output_spec.count)
        torch_save(sample_output_indices, self.sample_output_indices_tasks.file_name)

    def sample_output_file_name(self, save_point: int, index: int):
        return self.prefix + ("/sample_image_%03d_%03d.png" % (save_point, index))

    def reset_data_loader(self):
        self.training_data_loader = None
        self.training_data_loader_iter = None
        self.validation_data_loader = None
        self.validation_data_loader_iter = None

    def create_log_func(self, prefix: str, global_example_count: int):
        summary_writer = self.get_summary_writer()

        def log_func(tag: str, value: float):
            summary_writer.add_scalar(prefix + "_" + tag, value, global_example_count)

        return log_func

    def train(self, save_point: int):
        print("train(save_point = %d)" % save_point)

        print("load_rng_state")
        load_rng_state(self.rng_state_tasks.file_name(save_point - 1))

        print("load generator")
        G = self.load_generator(self.generator_tasks.file_name(save_point - 1))
        print("load generator")
        D = self.load_discriminator(self.discriminator_tasks.file_name(save_point - 1))
        print("load G_optim")
        G_optim = self.load_generator_optimizer(G, save_point - 1)
        print("load D_optim")
        D_optim = self.load_discriminator_optimizer(D, save_point - 1)

        print("set learning rate")
        set_learning_rate(G_optim, self.training_spec.generator_learning_rate(save_point - 1, 0))
        set_learning_rate(D_optim, self.training_spec.discriminator_learning_rate(save_point - 1, 0))

        print("reset data loader")
        self.reset_data_loader()

        batch_size = self.training_spec.batch_size
        save_point_example_count = 0
        global_example_count = (save_point - 1) * self.training_spec.example_seen_per_save_point
        validation_batch_index = 0
        sample_output_index = 0
        print("=== Training Save Point %d ===" % save_point)
        last_time = time.time()

        while save_point_example_count < self.training_spec.example_per_save_point:
            if save_point_example_count // self.validation_spec.example_per_batch >= validation_batch_index:
                validation_batch = self.get_next_validation_batch()
                G.train(False)
                D.train(False)
                log_func = self.create_log_func("validation", global_example_count)
                self.loss_spec.discriminator_loss(G, D, validation_batch, log_func)
                self.loss_spec.generator_loss(G, D, validation_batch, log_func)

            if save_point_example_count // self.sample_output_spec.example_per_sample_output >= sample_output_index:
                self.sample_output_spec.save_sample_image(
                    G,
                    self.get_sample_output_batch(),
                    self.sample_output_file_name(save_point, sample_output_index))
                sample_output_index += 1

            training_batch = self.get_next_training_batch()
            log_func = self.create_log_func("training", global_example_count)

            G_lr = self.training_spec.generator_learning_rate(save_point - 1, global_example_count)
            D_lr = self.training_spec.discriminator_learning_rate(save_point-1, global_example_count)
            set_learning_rate(G_optim, G_lr)
            set_learning_rate(D_optim, D_lr)
            self.get_summary_writer().add_scalar("generator_learning_rate", G_lr, global_example_count)
            self.get_summary_writer().add_scalar("discriminator_learning_rate", D_lr, global_example_count)

            if self.gan_spec.requires_discriminator_optimization():
                D.train(True)
                D.zero_grad()
                G.zero_grad()
                D_loss = self.loss_spec.discriminator_loss(G, D, training_batch, log_func)
                D_loss.backward()
                D_optim.step()

            if True:
                G.train(True)
                G.zero_grad()
                D.zero_grad()
                G_loss = self.loss_spec.generator_loss(G, D, training_batch, log_func)
                G_loss.backward()
                G_optim.step()

            save_point_example_count += batch_size
            global_example_count += batch_size
            now = time.time()
            if now - last_time > 10:
                print("Showed %d real images ..." % global_example_count)
                last_time = now

        print("done training")
        torch_save(G.state_dict(), self.generator_tasks.file_name(save_point))
        torch_save(D.state_dict(), self.discriminator_tasks.file_name(save_point))
        torch_save(G_optim.state_dict(), self.generator_optimizer_tasks.file_name(save_point))
        torch_save(D_optim.state_dict(), self.discriminator_optimizer_tasks.file_name(save_point))
        save_rng_state(self.rng_state_tasks.file_name(save_point))

        print("reset loader")
        self.discriminator_data_loader = None
        self.discriminator_data_loader_iter = None
        self.generator_data_loader = None
        self.generator_data_loader_iter = None


class SampleOutputIndicesTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="sample_output_indices",
            define_tasks_immediately=False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/sample_output_indices.pt"

    def create_file_task(self):
        self.workspace.create_file_task(
            self.file_name,
            [],
            lambda: self.gan_tasks.save_sample_output_indices())


class InitialGeneratorTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(gan_tasks.workspace, gan_tasks.prefix, "initial_generator", False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_generator.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.gan_tasks.save_initial_models())


class InitialDiscriminatorTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(gan_tasks.workspace,
                         gan_tasks.prefix,
                         "initial_discriminator",
                         False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_discriminator.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.gan_tasks.save_initial_models())


class InitialRngStateTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(gan_tasks.workspace,
                         gan_tasks.prefix,
                         "initial_rng_state",
                         False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_rng_state.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.gan_tasks.save_initial_models())


class RngStateTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="rng_state",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/rng_state_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.gan_tasks.save_point_dependencies(index),
            lambda: self.gan_tasks.process_save_point(index))


class GeneratorTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="generator",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.gan_tasks.save_point_dependencies(index),
            lambda: self.gan_tasks.process_save_point(index))


class DiscriminatorTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="discriminator",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.gan_tasks.save_point_dependencies(index),
            lambda: self.gan_tasks.process_save_point(index))


class GeneratorOptimizerTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="generator_optimizer",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_optimizer_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.gan_tasks.save_point_dependencies(index),
            lambda: self.gan_tasks.process_save_point(index))


class DiscriminatorOptimizerTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer2):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="discriminator_optimizer",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_optimizer_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.gan_tasks.save_point_dependencies(index),
            lambda: self.gan_tasks.process_save_point(index))
