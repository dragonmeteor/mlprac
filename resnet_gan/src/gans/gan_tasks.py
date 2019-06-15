import abc
import torch
from typing import Callable, List, Tuple

from torch.utils.data import DataLoader

from gans.gan_loss import GanLoss
from gans.gan_spec import Gan
from gans.util import torch_save, save_rng_state
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from pytasuku.indexed.one_index_file_tasks import OneIndexFileTasks

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
                 data_load_func: Callable[[int], DataLoader],
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

    def save_save_point_zero_files(self):
        pass

    def save_point_dependencies(self, save_point_index: int) -> List[str]:
        if save_point_index == 0:
            return [
                self.initial_rng_state_tasks.file_name,
                self.initial_generator_tasks.file_name,
                self.initial_discriminator_tasks.file_name,
                self.latent_vector_tasks.file_name]
        else:
            return [

            ]

    def train(self, save_point_index: int):
        pass


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
        pass


class DiscriminatorTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        pass


class GeneratorOptimizerTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        pass


class DiscriminatorOptimizerTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        pass

class GeneratorLossTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        pass

class DiscriminatorLossTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        pass


class GeneratorLossPlotTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        pass


class DiscriminatorLossPlotTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: GanTasks):
        pass