import abc
import os
import random
import time
from typing import List, Tuple, Callable, Iterable

import PIL.Image
import matplotlib.pyplot as plt
import numpy
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset

from hana.rindou.poser.v1.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v1.poser_gan_spec import PoserGanSpec
from hana.rindou.util import torch_save, save_rng_state, torch_load, load_rng_state, optimizer_to_device, \
    rgba_to_numpy_image_greenscreen
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from pytasuku.indexed.one_index_file_tasks import OneIndexFileTasks


class PoserGanTrainingSpec:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def save_point_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def example_per_save_point(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def generator_learning_rates(self) -> List[float]:
        pass

    @property
    @abc.abstractmethod
    def discriminator_learning_rates(self) -> List[float]:
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
    @abc.abstractmethod
    def random_seed(self) -> int:
        pass


class PoserGanSampleOutputSpec:
    __metaclass__ = abc.ABC

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


class PoserGanTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 gan_spec: PoserGanSpec,
                 loss_spec: PoserGanLoss,
                 training_dataloader_func: Callable[[int], Iterable[List[Tensor]]],
                 validation_dataset: Dataset,
                 training_spec: PoserGanTrainingSpec,
                 sample_output_spec: PoserGanSampleOutputSpec,
                 device=torch.device('cpu')):
        self.workspace = workspace
        self.prefix = prefix
        self.gan_spec = gan_spec
        self.loss_spec = loss_spec
        self.training_dataloader_func = training_dataloader_func
        self.validation_dataset = validation_dataset
        self.training_spec = training_spec
        self.sample_output_spec = sample_output_spec
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
        self.generator_loss_tasks = GeneratorLossTasks(self)
        self.discriminator_loss_tasks = DiscriminatorLossTasks(self)
        self.generator_loss_plot_tasks = GeneratorLossPlotTasks(self)
        self.discriminator_loss_plot_tasks = DiscriminatorLossPlotTasks(self)

        self.rng_state_tasks.define_tasks()
        self.generator_tasks.define_tasks()
        self.discriminator_tasks.define_tasks()
        self.generator_optimizer_tasks.define_tasks()
        self.discriminator_optimizer_tasks.define_tasks()
        self.generator_loss_tasks.define_tasks()
        self.discriminator_loss_tasks.define_tasks()
        self.generator_loss_plot_tasks.define_tasks()
        self.discriminator_loss_plot_tasks.define_tasks()

        # Loss plots
        loss_plot_deps = []
        for i in range(self.generator_loss_plot_tasks.count):
            loss_plot_deps.append(self.generator_loss_plot_tasks.file_name(i))
            loss_plot_deps.append(self.discriminator_loss_plot_tasks.file_name(i))
        self.workspace.create_command_task(self.prefix + "/loss_plot", loss_plot_deps)
        self.workspace.create_command_task(self.prefix + "/loss_plot_clean",
                                           [self.generator_loss_plot_tasks.clean_command,
                                            self.discriminator_loss_plot_tasks.clean_command])

        self.data_loader = None
        self.data_loader_iter = None
        self.sample_output_batch = None

    def save_initial_models(self):
        torch.manual_seed(self.training_spec.random_seed)

        generator = self.gan_spec.generator().to(self.device)
        torch_save(generator.state_dict(), self.initial_generator_tasks.file_name)

        discriminator = self.gan_spec.discriminator().to(self.device)
        torch_save(discriminator.state_dict(), self.initial_discriminator_tasks.file_name)

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
                self.discriminator_optimizer_tasks.file_name(save_point_index - 1),
                self.generator_loss_tasks.file_name(save_point_index - 1),
                self.discriminator_loss_tasks.file_name(save_point_index - 1)]

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
                       lr=self.training_spec.generator_learning_rates[0],
                       betas=self.training_spec.generator_betas)
        D_optim = Adam(D.parameters(),
                       lr=self.training_spec.discriminator_learning_rates[0],
                       betas=self.training_spec.discriminator_betas)
        generator_loss = [0.0]
        discriminator_loss = [0.0]

        torch_save(G.state_dict(), self.generator_tasks.file_name(0))
        torch_save(D.state_dict(), self.discriminator_tasks.file_name(0))
        torch_save(G_optim.state_dict(), self.generator_optimizer_tasks.file_name(0))
        torch_save(D_optim.state_dict(), self.discriminator_optimizer_tasks.file_name(0))
        torch_save(generator_loss, self.generator_loss_tasks.file_name(0))
        torch_save(discriminator_loss, self.discriminator_loss_tasks.file_name(0))
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
                       lr=self.training_spec.generator_learning_rates[save_point],
                       betas=self.training_spec.generator_betas)
        file_name = self.generator_optimizer_tasks.file_name(save_point)
        G_optim.load_state_dict(torch_load(file_name))
        optimizer_to_device(G_optim, self.device)
        return G_optim

    def load_discriminator_optimizer(self, D: Module, save_point: int):
        D_optim = Adam(D.parameters(),
                       lr=self.training_spec.discriminator_learning_rates[save_point],
                       betas=self.training_spec.discriminator_betas)
        file_name = self.discriminator_optimizer_tasks.file_name(save_point)
        D_optim.load_state_dict(torch_load(file_name))
        optimizer_to_device(D_optim, self.device)
        return D_optim

    def get_next_example_batch(self):
        if self.data_loader is None:
            self.data_loader = self.training_dataloader_func(self.training_spec.batch_size)
        if self.data_loader_iter is None:
            self.data_loader_iter = iter(self.data_loader)
        try:
            batch = next(self.data_loader_iter)
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            batch = next(self.data_loader_iter)
        return [
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device)
        ]

    def get_sample_output_batch(self):
        if self.sample_output_batch is not None:
            return self.sample_output_batch

        source_images = []
        poses = []
        target_images = []
        sample_output_indices = torch_load(self.sample_output_indices_tasks.file_name)
        for index in sample_output_indices:
            source_image, pose, target_image = self.validation_dataset[index]
            source_image = source_image.to(self.device).unsqueeze(0)
            pose = pose.to(self.device).unsqueeze(0)
            target_image = target_image.to(self.device).unsqueeze(0)
            source_images.append(source_image)
            poses.append(pose)
            target_images.append(target_image)
        self.sample_output_batch = [
            torch.cat(source_images, dim=0),
            torch.cat(poses, dim=0),
            torch.cat(target_images, dim=0)
        ]

        return self.sample_output_batch

    def save_sample_output_indices(self):
        n = len(self.validation_dataset)
        random.seed(self.sample_output_spec.sample_output_index_seed)
        sample_output_indices = random.sample(range(n), self.sample_output_spec.count)
        torch_save(sample_output_indices, self.sample_output_indices_tasks.file_name)

    def save_sample_output(self, G: Module, file_name: str):
        G.train(False)

        sample_output_batch = self.get_sample_output_batch()
        source_image = sample_output_batch[0]
        pose = sample_output_batch[1]
        target_image = sample_output_batch[2]

        output = G(source_image, pose)
        self.save_output_image(source_image, target_image, output[0].detach(), file_name)

    def save_output_image(self, source_images, target_images, output_images, file_name):
        source_images = F.interpolate(source_images, size=self.sample_output_spec.image_size).cpu()
        target_images = F.interpolate(target_images, size=self.sample_output_spec.image_size).cpu()
        output_images = F.interpolate(output_images, size=self.sample_output_spec.image_size).cpu()

        n = output_images.shape[0]
        num_rows = n // self.sample_output_spec.example_per_row
        if n % self.sample_output_spec.example_per_row != 0:
            num_rows += 1
        num_cols = 3 * self.sample_output_spec.example_per_row

        image_size = self.sample_output_spec.image_size
        output = numpy.zeros([num_rows * image_size, num_cols * image_size, 3])
        for i in range(n):
            row = i // self.sample_output_spec.example_per_row
            col = i % self.sample_output_spec.example_per_row

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

    def sample_output_file_name(self, save_point: int, index: int):
        return self.prefix + ("/sample_image_%03d_%03d.png" % (save_point, index))

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
        print("load generator loss")
        generator_loss = torch_load(self.generator_loss_tasks.file_name(save_point - 1))
        print("load discriminator loss")
        discriminator_loss = torch_load(self.discriminator_loss_tasks.file_name(save_point - 1))

        print("set learning rate")
        for param_group in G_optim.param_groups:
            param_group['lr'] = self.training_spec.generator_learning_rates[save_point - 1]
        for param_group in D_optim.param_groups:
            param_group['lr'] = self.training_spec.discriminator_learning_rates[save_point - 1]

        print("reset loader")
        self.discriminator_data_loader = None
        self.discriminator_data_loader_iter = None
        self.generator_data_loader = None
        self.generator_data_loader_iter = None
        example_count = 0

        batch_size = self.training_spec.batch_size
        iter_index = 0
        sample_output_index = 0
        print("=== Training Save Point %d ===" % save_point)
        last_time = time.time()

        while example_count < self.training_spec.example_per_save_point:
            if example_count // self.sample_output_spec.example_per_sample_output >= sample_output_index:
                self.save_sample_output(G, self.sample_output_file_name(save_point, sample_output_index))
                sample_output_index += 1

            batch = self.get_next_example_batch()

            if self.gan_spec.requires_discriminator_optimization():
                D.train(True)
                D.zero_grad()
                G.zero_grad()
                D_loss = self.loss_spec.discriminator_loss(G, D, batch)
                D_loss.backward()
                D_optim.step()
                discriminator_loss.append(D_loss.item())

            if True:
                G.train(True)
                G.zero_grad()
                D.zero_grad()
                G_loss = self.loss_spec.generator_loss(G, D, batch)
                G_loss.backward()
                G_optim.step()
                generator_loss.append(G_loss.item())

            example_count += batch_size

            iter_index += 1
            now = time.time()
            if now - last_time > 10:
                print("Showed %d real images ..." % (iter_index * batch_size))
                last_time = now

        print("done training")
        torch_save(G.state_dict(), self.generator_tasks.file_name(save_point))
        torch_save(D.state_dict(), self.discriminator_tasks.file_name(save_point))
        torch_save(G_optim.state_dict(), self.generator_optimizer_tasks.file_name(save_point))
        torch_save(D_optim.state_dict(), self.discriminator_optimizer_tasks.file_name(save_point))
        torch_save(generator_loss, self.generator_loss_tasks.file_name(save_point))
        torch_save(discriminator_loss, self.discriminator_loss_tasks.file_name(save_point))
        save_rng_state(self.rng_state_tasks.file_name(save_point))

        print("reset loader")
        self.discriminator_data_loader = None
        self.discriminator_data_loader_iter = None
        self.generator_data_loader = None
        self.generator_data_loader_iter = None


class SampleOutputIndicesTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasks):
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
    def __init__(self, gan_tasks: PoserGanTasks):
        super().__init__(gan_tasks.workspace, gan_tasks.prefix, "initial_generator", False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_generator.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.gan_tasks.save_initial_models())


class InitialDiscriminatorTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasks):
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
    def __init__(self, gan_tasks: PoserGanTasks):
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
    def __init__(self, gan_tasks: PoserGanTasks):
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
    def __init__(self, gan_tasks: PoserGanTasks):
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
    def __init__(self, gan_tasks: PoserGanTasks):
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
    def __init__(self, gan_tasks: PoserGanTasks):
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
    def __init__(self, gan_tasks: PoserGanTasks):
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


class GeneratorLossTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasks):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="generator_loss",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_loss_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.gan_tasks.save_point_dependencies(index),
            lambda: self.gan_tasks.process_save_point(index))


class DiscriminatorLossTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasks):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="discriminator_loss",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_loss_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.gan_tasks.save_point_dependencies(index),
            lambda: self.gan_tasks.process_save_point(index))


def plot_loss_(loss, title, y_label, file_name, window_size=100):
    plt.figure()
    if len(loss) == 1:
        plt.plot(loss)
    else:
        raw_loss = loss[1:]
        loss = []
        sum = 0
        for i in range(1, len(raw_loss)):
            sum += raw_loss[i]
            if i - window_size >= 0:
                sum -= raw_loss[i - window_size]
            mean = sum * 1.0 / min(window_size, (i + 1))
            loss.append(mean)
        plt.plot(loss)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_name, format='png')
    plt.close()


class GeneratorLossPlotTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasks):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="generator_loss_plot",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_loss_plot_%03d.png" % index)

    def plot_loss(self, index):
        loss = torch_load(self.gan_tasks.generator_loss_tasks.file_name(index))
        title = "Generator Loss (save_point=%d)" % index
        plot_loss_(loss, title, "Loss", self.file_name(index))

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            [self.gan_tasks.generator_loss_tasks.file_name(index)],
            lambda: self.plot_loss(index))


class DiscriminatorLossPlotTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasks):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="discriminator_loss_plot",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_loss_plot_%03d.png" % index)

    def plot_loss(self, index):
        loss = torch_load(self.gan_tasks.discriminator_loss_tasks.file_name(index))
        title = "Discriminator Loss (save_point=%d)" % index
        plot_loss_(loss, title, "Loss", self.file_name(index))

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            [self.gan_tasks.discriminator_loss_tasks.file_name(index)],
            lambda: self.plot_loss(index))
