import random
import shutil
import time
from datetime import datetime
from typing import List

import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from hana.rindou.poser.v2.poser_gan_loss import PoserGanLoss
from hana.rindou.poser.v1.poser_gan_tasks_ver2 import PoserGanTrainingSpecVer2, PoserGanValidationSpecVer2, \
    PoserGanSampleOutputSpecVer2
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec
from hana.rindou.util import torch_save, save_rng_state, torch_load, load_rng_state, optimizer_to_device
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from pytasuku.indexed.one_index_file_tasks import OneIndexFileTasks


def set_learning_rate(module, lr):
    for param_group in module.param_groups:
        param_group['lr'] = lr


class PoserGanTasksVer3:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 generator_spec: PoserGanModuleSpec,
                 discriminator_spec: PoserGanModuleSpec,
                 generator_loss: PoserGanLoss,
                 discriminator_loss: PoserGanLoss,
                 training_dataset: Dataset,
                 validation_dataset: Dataset,
                 training_spec: PoserGanTrainingSpecVer2,
                 validation_spec: PoserGanValidationSpecVer2,
                 sample_output_spec: PoserGanSampleOutputSpecVer2,
                 pretrained_generator_file_name: str = None,
                 pretrained_discriminator_file_name: str = None,
                 device=torch.device('cpu'),
                 num_data_loader_workers: int = 4,
                 perform_validation: bool = True,
                 save_sample_output: bool = True):
        self.workspace = workspace
        self.prefix = prefix
        self.generator_spec = generator_spec
        self.discriminator_spec = discriminator_spec
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.training_spec = training_spec
        self.validation_spec = validation_spec
        self.sample_output_spec = sample_output_spec
        self.pretrained_generator_file_name = pretrained_generator_file_name
        self.pretrained_discriminator_file_name = pretrained_discriminator_file_name
        self.num_data_loader_workers = num_data_loader_workers
        self.perform_validation = perform_validation
        self.save_sample_output = save_sample_output
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
            generator = self.generator_spec.get_module().to(self.device)
            torch_save(generator.state_dict(), self.initial_generator_tasks.file_name)
        else:
            shutil.copyfile(self.pretrained_generator_file_name, self.initial_generator_tasks.file_name)

        if self.pretrained_discriminator_file_name is None:
            discriminator = self.discriminator_spec.get_module().to(self.device)
            torch_save(discriminator.state_dict(), self.initial_discriminator_tasks.file_name)
        else:
            shutil.copyfile(self.pretrained_discriminator_file_name, self.initial_discriminator_tasks.file_name)

        save_rng_state(self.initial_rng_state_tasks.file_name)

    def initial_dependencies(self) -> List[str]:
        dependencies = []
        if self.pretrained_discriminator_file_name is not None:
            dependencies.append(self.pretrained_discriminator_file_name)
        if self.pretrained_generator_file_name is not None:
            dependencies.append(self.pretrained_generator_file_name)
        return dependencies

    def save_point_dependencies(self, save_point_index: int) -> List[str]:
        if save_point_index == 0:
            dependencies = [
                self.initial_rng_state_tasks.file_name,
                self.initial_generator_tasks.file_name,
                self.initial_discriminator_tasks.file_name,
                self.sample_output_indices_tasks.file_name,
            ]
            if self.pretrained_discriminator_file_name is not None:
                dependencies.append(self.pretrained_discriminator_file_name)
            if self.pretrained_generator_file_name is not None:
                dependencies.append(self.pretrained_generator_file_name)
            return dependencies
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
        G = self.generator_spec.get_module().to(self.device)
        G.load_state_dict(torch_load(file_name), strict=False)
        return G

    def load_discriminator(self, file_name):
        D = self.discriminator_spec.get_module().to(self.device)
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
                num_workers=self.num_data_loader_workers,
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
                num_workers=self.num_data_loader_workers,
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
            if self.perform_validation \
                    and save_point_example_count // self.validation_spec.example_per_batch >= validation_batch_index:
                validation_batch = self.get_next_validation_batch()
                G.train(False)
                D.train(False)
                if self.discriminator_loss is not None:
                    log_func = self.create_log_func("validation_discriminator", global_example_count)
                    self.discriminator_loss.compute(G, D, validation_batch, log_func)
                if self.generator_loss is not None:
                    log_func = self.create_log_func("validation_generator", global_example_count)
                    self.generator_loss.compute(G, D, validation_batch, log_func)

            if self.save_sample_output \
                    and save_point_example_count // self.sample_output_spec.example_per_sample_output >= sample_output_index:
                self.sample_output_spec.save_sample_image(
                    G,
                    self.get_sample_output_batch(),
                    self.sample_output_file_name(save_point, sample_output_index))
                sample_output_index += 1

            training_batch = self.get_next_training_batch()

            G_lr = self.training_spec.generator_learning_rate(save_point - 1, global_example_count)
            D_lr = self.training_spec.discriminator_learning_rate(save_point - 1, global_example_count)
            set_learning_rate(G_optim, G_lr)
            set_learning_rate(D_optim, D_lr)
            self.get_summary_writer().add_scalar("generator_learning_rate", G_lr, global_example_count)
            self.get_summary_writer().add_scalar("discriminator_learning_rate", D_lr, global_example_count)

            if self.discriminator_spec.requires_optimization():
                D.train(True)
                D.zero_grad()
                G.zero_grad()
                log_func = self.create_log_func("training_discriminator", global_example_count)
                D_loss = self.discriminator_loss.compute(G, D, training_batch, log_func)
                D_loss.backward()
                D_optim.step()

            if self.generator_spec.requires_optimization():
                G.train(True)
                G.zero_grad()
                D.zero_grad()
                log_func = self.create_log_func("training_generator", global_example_count)
                G_loss = self.generator_loss.compute(G, D, training_batch, log_func)
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
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
    def __init__(self, gan_tasks: PoserGanTasksVer3):
        super().__init__(gan_tasks.workspace, gan_tasks.prefix, "initial_generator", False)
        self.gan_tasks = gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_generator.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name,
                                        self.gan_tasks.initial_dependencies(),
                                        lambda: self.gan_tasks.save_initial_models())


class InitialDiscriminatorTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
        self.workspace.create_file_task(self.file_name,
                                        self.gan_tasks.initial_dependencies(),
                                        lambda: self.gan_tasks.save_initial_models())


class InitialRngStateTasks(NoIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
        self.workspace.create_file_task(self.file_name,
                                        self.gan_tasks.initial_dependencies(),
                                        lambda: self.gan_tasks.save_initial_models())


class RngStateTasks(OneIndexFileTasks):
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
    def __init__(self, gan_tasks: PoserGanTasksVer3):
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
