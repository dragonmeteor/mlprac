import abc
import time

import torch
from typing import Callable, List, Tuple

from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from gans.gan_loss import GanLoss
from gans.gan_spec import Gan
from gans.util import torch_save, save_rng_state, torch_load, load_rng_state, optimizer_to_device, save_sample_images
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from pytasuku.indexed.one_index_file_tasks import OneIndexFileTasks

import matplotlib.pyplot as plt

DEFAULT_BATCH_SIZE = 16


class GanTrainingSpec:
    __metaclass__ = abc.ABC

    @property
    @abc.abstractmethod
    def save_point_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def sample_per_save_point(self) -> int:
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
    def random_seed(self) -> int:
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
        output = self.sample_per_save_point // self.batch_size
        if self.sample_per_save_point % self.batch_size != 0:
            output += 1
        return output


class SampleImageSpec:
    __metaclass__ = abc.ABC

    @property
    @abc.abstractmethod
    def count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def image_per_row(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def sample_per_sample_image(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def latent_vector_seed(self) -> int:
        pass


class GanTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 gan_spec: Gan,
                 loss_spec: GanLoss,
                 data_load_func: Callable[[int, torch.device], DataLoader],
                 training_spec: GanTrainingSpec,
                 sample_image_spec: SampleImageSpec,
                 device=torch.device('cpu')):
        self.workspace = workspace
        self.prefix = prefix
        self.gan_spec = gan_spec
        self.loss_spec = loss_spec
        self.data_load_func = data_load_func
        self.training_spec = training_spec
        self.sample_image_spec = sample_image_spec
        self.device = device

        # Random fixed inputs for generating example images.
        self.latent_vector_tasks = LatentVectorFileTasks(self)

        # Initial models.
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

        self.discriminator_data_loader = None
        self.discriminator_data_loader_iter = None
        self.generator_data_loader = None
        self.generator_data_loader_iter = None

    def sample_latent_vectors(self, count):
        return torch.randn(
            count,
            self.gan_spec.latent_vector_size,
            device=self.device)

    def save_initial_models(self):
        torch.manual_seed(self.training_spec.random_seed)

        generator = self.gan_spec.generator().to(self.device)
        torch_save(generator.state_dict(), self.initial_generator_tasks.file_name)

        discriminator = self.gan_spec.discriminator().to(self.device)
        torch_save(discriminator.state_dict(), self.initial_discriminator_tasks.file_name)

        save_rng_state(self.initial_rng_state_tasks.file_name)

    def process_save_point(self, save_point_index: int):
        if save_point_index == 0:
            self.save_save_point_zero_files()
        else:
            self.train(save_point_index)

    def save_point_dependencies(self, save_point_index: int) -> List[str]:
        if save_point_index == 0:
            return [
                self.initial_rng_state_tasks.file_name,
                self.initial_generator_tasks.file_name,
                self.initial_discriminator_tasks.file_name,
                self.latent_vector_tasks.file_name]
        else:
            return [
                self.rng_state_tasks.file_name(save_point_index - 1),
                self.generator_tasks.file_name(save_point_index - 1),
                self.discriminator_tasks.file_name(save_point_index - 1),
                self.generator_optimizer_tasks.file_name(save_point_index - 1),
                self.discriminator_optimizer_tasks.file_name(save_point_index - 1),
                self.generator_loss_tasks.file_name(save_point_index - 1),
                self.discriminator_loss_tasks.file_name(save_point_index - 1),
                self.generator_loss_plot_tasks.file_name(save_point_index - 1),
                self.discriminator_loss_plot_tasks.file_name(save_point_index - 1),
                self.latent_vector_tasks.file_name]

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

    def sample_image_file_name(self, save_point: int, index: int):
        return self.prefix + ("/sample_image_%03d_%03d.png" % (save_point, index))

    def load_latent_vector(self):
        return torch_load(self.latent_vector_tasks.file_name)

    def save_sample_images(self,
                           generator: Module,
                           batch_size: int,
                           file_name: str):
        latent_vector = self.load_latent_vector()
        self.save_sample_images_from_input_data(generator,
                                                latent_vector,
                                                batch_size,
                                                file_name)

    def save_sample_images_from_input_data(self,
                                           generator: Module,
                                           latent_vector: torch.Tensor,
                                           batch_size: int,
                                           file_name: str):
        generator.train(False)

        sample_images = None
        while (sample_images is None) or (sample_images.shape[0] < self.sample_image_spec.count):
            if sample_images is None:
                vectors = latent_vector[:batch_size]
            else:
                limit = max(batch_size, self.sample_image_spec.count - sample_images.shape[0])
                vectors = latent_vector[sample_images.shape[0]:sample_images.shape[0] + limit]
            images = generator(vectors).detach()
            if sample_images is None:
                sample_images = images
            else:
                sample_images = torch.cat((sample_images, images), dim=0)
        save_sample_images(sample_images.detach().cpu(),
                           self.gan_spec.image_size,
                           self.sample_image_spec.image_per_row,
                           file_name)
        print("Saved %s" % file_name)

    def get_discriminator_next_real_image_batch(self) -> torch.Tensor:
        if self.discriminator_data_loader is None:
            self.discriminator_data_loader = self.data_load_func(self.training_spec.batch_size, self.device)
        if self.discriminator_data_loader_iter is None:
            self.discriminator_data_loader_iter = self.discriminator_data_loader.__iter__()
        try:
            output = self.discriminator_data_loader_iter.__next__()[0]
        except StopIteration:
            self.discriminator_data_loader_iter = self.discriminator_data_loader.__iter__()
            output = self.discriminator_data_loader_iter.__next__()[0]
        return output

    def get_generator_next_real_image_batch(self) -> torch.Tensor:
        if self.generator_data_loader is None:
            self.generator_data_loader = self.data_load_func(self.training_spec.batch_size, self.device)
        if self.generator_data_loader_iter is None:
            self.generator_data_loader_iter = self.generator_data_loader.__iter__()
        try:
            output = self.generator_data_loader_iter.__next__()[0]
        except StopIteration:
            self.generator_data_loader_iter = self.generator_data_loader.__iter__()
            output = self.generator_data_loader_iter.__next__()[0]
        return output

    def train(self, save_point: int):
        load_rng_state(self.rng_state_tasks.file_name(save_point - 1))

        G = self.load_generator(self.generator_tasks.file_name(save_point - 1))
        D = self.load_discriminator(self.discriminator_tasks.file_name(save_point - 1))
        G_optim = self.load_generator_optimizer(G, save_point - 1)
        D_optim = self.load_discriminator_optimizer(D, save_point - 1)
        generator_loss = torch_load(self.generator_loss_tasks.file_name(save_point - 1))
        discriminator_loss = torch_load(self.discriminator_loss_tasks.file_name(save_point - 1))

        self.discriminator_data_loader = None
        self.discriminator_data_loader_iter = None
        self.generator_data_loader = None
        self.generator_data_loader_iter = None
        sample_count = 0

        batch_size = self.training_spec.batch_size
        sample_image_index = 0
        iter_index = 0
        print("=== Training Save Point %d ===" % save_point)
        last_time = time.time()
        alpha = 0.0

        while sample_count < self.training_spec.sample_per_save_point:
            if sample_count / self.sample_image_spec.sample_per_sample_image >= sample_image_index:
                self.save_sample_images(
                    generator=G,
                    batch_size=self.training_spec.batch_size,
                    file_name=self.sample_image_file_name(save_point,
                                                          sample_image_index))
                sample_image_index += 1

            if True:
                real_images = self.get_discriminator_next_real_image_batch()
                latent_vectors = self.sample_latent_vectors(batch_size)
                D.train(True)
                D.zero_grad()
                G.zero_grad()
                D_loss = self.loss_spec.discriminator_loss(G, D, real_images, latent_vectors)
                D_loss.backward()
                D_optim.step()

                discriminator_loss.append(D_loss.item())

            if True:
                real_images = self.get_generator_next_real_image_batch()
                latent_vectors = self.sample_latent_vectors(batch_size)
                G.train(True)
                G.zero_grad()
                D.zero_grad()
                G_loss = self.loss_spec.generator_loss(G, D, real_images, latent_vectors)
                G_loss.backward()
                G_optim.step()

                generator_loss.append(G_loss.item())

            sample_count += batch_size

            iter_index += 1
            now = time.time()
            if now - last_time > 10:
                print("Showed %d real images ..." % (iter_index * batch_size))

        torch_save(G.state_dict(), self.generator_tasks.file_name(save_point))
        torch_save(D.state_dict(), self.discriminator_tasks.file_name(save_point))
        torch_save(G_optim.state_dict(), self.generator_optimizer_tasks.file_name(save_point))
        torch_save(D_optim.state_dict(), self.discriminator_optimizer_tasks.file_name(save_point))
        torch_save(generator_loss, self.generator_loss_tasks.file_name(save_point))
        torch_save(discriminator_loss, self.discriminator_loss_tasks.file_name(save_point))
        save_rng_state(self.rng_state_tasks.file_name(save_point))

        self.discriminator_data_loader = None
        self.discriminator_data_loader_iter = None
        self.generator_data_loader = None
        self.generator_data_loader_iter = None


class LatentVectorFileTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        super().__init__(
            gan_tasks.workspace,
            gan_tasks.prefix,
            "latent_vector",
            False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/latent_vector.pt"

    def save_latent_vectors(self):
        torch.manual_seed(self.gan_tasks.sample_image_spec.latent_vector_seed)
        latent_vectors = self.gan_tasks.sample_latent_vectors(
            self.gan_tasks.sample_image_spec.count)
        torch_save(latent_vectors, self.file_name)

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.save_latent_vectors())


class InitialGeneratorTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        super().__init__(gan_tasks.workspace,
                         gan_tasks.prefix,
                         "initial_generator_module",
                         False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_generator_module.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.gan_tasks.save_initial_models())


class InitialDiscriminatorTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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
    def __init__(self, gan_tasks: GanTasks):
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


def plot_loss(loss, title, y_label, file_name, window_size=100):
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
    def __init__(self, gan_tasks: GanTasks):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="generator_loss_plot",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_loss_plot_%03d.pt" % index)

    def plot_loss(self, index):
        loss = torch_load(self.gan_tasks.generator_loss_tasks.file_name(index))
        title = "Generator Loss (save_point=%d)" % index
        plot_loss(loss, title, "Loss", self.file_name(index))

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            [self.gan_tasks.generator_loss_tasks.file_name(index)],
            lambda: self.plot_loss(index))


class DiscriminatorLossPlotTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        super().__init__(
            workspace=gan_tasks.workspace,
            prefix=gan_tasks.prefix,
            command_name="discriminator_loss_plot",
            count=gan_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.gan_tasks = gan_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_loss_plot_%03d.pt" % index)

    def plot_loss(self, index):
        loss = torch_load(self.gan_tasks.discriminator_loss_tasks.file_name(index))
        title = "Discriminator Loss (save_point=%d)" % index
        plot_loss(loss, title, "Loss", self.file_name(index))

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            [self.gan_tasks.discriminator_loss_tasks.file_name(index)],
            lambda: self.plot_loss(index))
