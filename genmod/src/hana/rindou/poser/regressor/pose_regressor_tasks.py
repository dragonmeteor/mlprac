import abc
import math
from datetime import datetime
import time
from typing import Tuple

import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from hana.rindou.poser.regressor.pose_regressor_loss import PoseRegressorLoss
from hana.rindou.poser.regressor.pose_regressor_spec import PoseRegressorSpec
from hana.rindou.util import torch_save, save_rng_state, load_rng_state, torch_load, optimizer_to_device
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from pytasuku.indexed.one_index_file_tasks import OneIndexFileTasks


class PoseRegressorTrainingSpec:
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
    def learning_rate(self, save_point_index: int, example_seen_so_far: int) -> float:
        pass

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def betas(self) -> Tuple[float, float]:
        pass

    @property
    @abc.abstractmethod
    def random_seed(self) -> int:
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


class PoseRegressorValidationSpec:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def batch_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def example_per_batch(self) -> int:
        pass


class PoseRegressorTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 pose_regressor_spec: PoseRegressorSpec,
                 loss: PoseRegressorLoss,
                 training_spec: PoseRegressorTrainingSpec,
                 training_dataset: Dataset,
                 validation_spec: PoseRegressorValidationSpec,
                 validation_dataset: Dataset,
                 device=torch.device('cpu')):
        self.workspace = workspace
        self.prefix = prefix
        self.pose_regressor_spec = pose_regressor_spec
        self.loss = loss
        self.training_spec = training_spec
        self.training_dataset = training_dataset
        self.validation_spec = validation_spec
        self.validation_dataset = validation_dataset
        self.device = device
        self.log_dir = None

        # Initial models
        self.initial_regressor_tasks = InitialRegressorTasks(self)
        self.initial_rng_state_tasks = InitialRngStateTasks(self)

        # Training tasks
        self.rng_state_tasks = RngStateTasks(self)
        self.regressor_tasks = RegressorTasks(self)
        self.regressor_optimizer_tasks = RegressorOptimizerTasks(self)

        self.rng_state_tasks.define_tasks()
        self.regressor_tasks.define_tasks()
        self.regressor_optimizer_tasks.define_tasks()

        self.learning_rate = None

    def get_log_dir(self):
        if self.log_dir is None:
            now = datetime.now()
            self.log_dir = self.prefix + "/log/" + now.strftime("%Y_%m_%d__%H_%M_%S")
        return self.log_dir


    def save_initial_models(self):
        torch.manual_seed(self.training_spec.random_seed)

        R = self.pose_regressor_spec.regressor().to(self.device)
        torch_save(R.state_dict(), self.initial_regressor_tasks.file_name)

        save_rng_state(self.initial_rng_state_tasks.file_name)

    def save_point_dependencies(self, save_point_index):
        if save_point_index == 0:
            return [
                self.initial_rng_state_tasks.file_name,
                self.initial_regressor_tasks.file_name,
            ]
        else:
            return [
                self.rng_state_tasks.file_name(save_point_index - 1),
                self.regressor_tasks.file_name(save_point_index - 1),
                self.regressor_optimizer_tasks.file_name(save_point_index - 1),
            ]

    def process_save_point(self, save_point_index):
        if save_point_index == 0:
            self.save_save_point_zero_files()
        else:
            self.train(save_point_index)

    def save_save_point_zero_files(self):
        load_rng_state(self.initial_rng_state_tasks.file_name)
        R = self.load_regressor(self.initial_regressor_tasks.file_name)
        R_optim = Adam(R.parameters(),
                       lr=self.training_spec.learning_rate(0, 0),
                       betas=self.training_spec.betas)

        torch_save(R.state_dict(), self.regressor_tasks.file_name(0))
        torch_save(R_optim.state_dict(), self.regressor_optimizer_tasks.file_name(0))
        save_rng_state(self.rng_state_tasks.file_name(0))

    def load_regressor(self, file_name):
        R = self.pose_regressor_spec.regressor().to(self.device)
        R.load_state_dict(torch_load(file_name))
        return R

    def load_regressor_optimizer(self, R: Module, save_point: int):
        R_optim = Adam(R.parameters(),
                       lr=self.training_spec.learning_rate(save_point, 0),
                       betas=self.training_spec.betas)
        file_name = self.regressor_optimizer_tasks.file_name(save_point)
        R_optim.load_state_dict(torch_load(file_name))
        optimizer_to_device(R_optim, self.device)
        return R_optim

    def set_learning_rate(self, R_optim, new_learning_rate):
        if self.learning_rate != new_learning_rate:
            print("set learning rate to", new_learning_rate)
            for param_group in R_optim.param_groups:
                param_group['lr'] = new_learning_rate
            self.learning_rate = new_learning_rate

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
            self.validation_data_loader_iter = iter(self.validation_data_loader)
            batch = next(self.validation_data_loader_iter)
        return [x.to(self.device) for x in batch]

    def train(self, save_point: int):
        print("train(save_point = %d)" % save_point)

        print("load_rng_state")
        load_rng_state(self.rng_state_tasks.file_name(save_point - 1))
        print("load regressor")
        R = self.load_regressor(self.regressor_tasks.file_name(save_point - 1))
        print("load regressor optimizer")
        R_optim = self.load_regressor_optimizer(R, save_point - 1)

        print("reset loader")
        self.training_data_loader = None
        self.training_data_loader_iter = None
        self.validation_data_loader = None
        self.validation_data_loader_iter = None

        batch_size = self.training_spec.batch_size
        save_point_example_count = 0
        global_example_count = (save_point - 1) * self.training_spec.example_seen_per_save_point
        validation_batch_index = 0
        print("=== Training Save Point %d ===" % save_point)
        last_time = time.time()
        summary_writer = SummaryWriter(log_dir=self.get_log_dir())

        while save_point_example_count < self.training_spec.example_per_save_point:
            if save_point_example_count // self.validation_spec.example_per_batch >= validation_batch_index:
                batch = self.get_next_validation_batch()
                R.train(False)
                R_loss = self.loss.compute(R, batch)
                summary_writer.add_scalar("validation_loss", R_loss.item(), global_example_count)
                validation_batch_index += 1

            lr = self.training_spec.learning_rate(save_point - 1, global_example_count)
            self.set_learning_rate(R_optim, lr)
            summary_writer.add_scalar("learning_rate", lr, global_example_count)

            batch = self.get_next_training_batch()
            R.train(True)
            R.zero_grad()
            R_loss = self.loss.compute(R, batch)
            R_loss.backward()
            R_optim.step()
            summary_writer.add_scalar("training_loss", R_loss.item(), global_example_count)

            save_point_example_count += batch_size
            global_example_count += batch_size
            now = time.time()
            if now - last_time > 10:
                print("Showed %d examples ..." % global_example_count)
                last_time = now

        print("done training")
        torch_save(R.state_dict(), self.regressor_tasks.file_name(save_point))
        torch_save(R_optim.state_dict(), self.regressor_optimizer_tasks.file_name(save_point))
        save_rng_state(self.rng_state_tasks.file_name(save_point))


class InitialRegressorTasks(NoIndexFileTasks):
    def __init__(self, regressor_tasks: PoseRegressorTasks):
        super().__init__(
            regressor_tasks.workspace,
            regressor_tasks.prefix,
            "initial_regressor",
            False)
        self.regressor_tasks = regressor_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_regressor.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.regressor_tasks.save_initial_models())


class InitialRngStateTasks(NoIndexFileTasks):
    def __init__(self, regressor_tasks: PoseRegressorTasks):
        super().__init__(
            regressor_tasks.workspace,
            regressor_tasks.prefix,
            "initial_rng_state",
            False)
        self.regressor_tasks = regressor_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_rng_state.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.regressor_tasks.save_initial_models())


class RngStateTasks(OneIndexFileTasks):
    def __init__(self, regressor_tasks: PoseRegressorTasks):
        super().__init__(
            workspace=regressor_tasks.workspace,
            prefix=regressor_tasks.prefix,
            command_name="rng_state",
            count=regressor_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.regressor_tastks = regressor_tasks

    def file_name(self, index):
        return self.prefix + ("/rng_state_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.regressor_tastks.save_point_dependencies(index),
            lambda: self.regressor_tastks.process_save_point(index))


class RegressorTasks(OneIndexFileTasks):
    def __init__(self, regressor_tasks: PoseRegressorTasks):
        super().__init__(
            workspace=regressor_tasks.workspace,
            prefix=regressor_tasks.prefix,
            command_name="regressor",
            count=regressor_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.regressor_tastks = regressor_tasks

    def file_name(self, index):
        return self.prefix + ("/regressor_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.regressor_tastks.save_point_dependencies(index),
            lambda: self.regressor_tastks.process_save_point(index))


class RegressorOptimizerTasks(OneIndexFileTasks):
    def __init__(self, regressor_tasks: PoseRegressorTasks):
        super().__init__(
            workspace=regressor_tasks.workspace,
            prefix=regressor_tasks.prefix,
            command_name="regressor_optimizer",
            count=regressor_tasks.training_spec.save_point_count + 1,
            define_tasks_at_creation=False)
        self.regressor_tastks = regressor_tasks

    def file_name(self, index):
        return self.prefix + ("/regressor_optimizer_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(
            self.file_name(index),
            self.regressor_tastks.save_point_dependencies(index),
            lambda: self.regressor_tastks.process_save_point(index))
