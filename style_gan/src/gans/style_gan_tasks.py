import time
from typing import Callable, Dict, List

import shutil
import os
import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from gans.gan_loss import GanLoss
from gans.style_gan_spec import StyleGan
from gans.util import torch_save, save_sample_images, torch_load, save_rng_state, load_rng_state, optimizer_to_device
from pytasuku import Workspace
from pytasuku.indexed.no_index_file_tasks import NoIndexFileTasks
from pytasuku.indexed.one_index_file_tasks import OneIndexFileTasks
from pytasuku.indexed.two_indices_file_tasks import TwoIndicesFileTasks

DEFAULT_BATCH_SIZE = {
    4: 32,
    8: 32,
    16: 32,
    32: 32,
    64: 16,
    128: 16,
    256: 16,
    512: 8,
    1024: 8
}

STABILIZE_PHASE_NAME = "stabilize"
TRANSITION_PHASE_NAME = "transition"


class StyleGanTasks:
    def __init__(self,
                 workspace: Workspace,
                 prefix: str,
                 style_gan_spec: StyleGan,
                 loss_spec: GanLoss,
                 output_image_size: int,
                 data_loader_func: Callable[[int, int, torch.device], DataLoader],
                 latent_vector_seed=293404984,
                 training_seed=60586483,
                 batch_size: Dict[int, int] = None,
                 sample_image_count=64,
                 sample_image_per_row=8,
                 sample_per_sample_image=10000,
                 sample_per_loss_record=1000,
                 sample_per_save_point=100000,
                 save_point_per_phase=6,
                 discriminator_learning_rate=1e-4,
                 generator_module_learning_rate=1e-4,
                 mapping_module_learning_rate=1e-6,
                 generator_module_betas=(0, 0.999),
                 discriminator_betas=(0, 0.999),
                 mapping_module_betas=(0, 0.999),
                 device=torch.device('cpu')):
        self.workspace = workspace
        self.prefix = prefix
        self.style_gan_spec = style_gan_spec
        self.loss_spec = loss_spec
        self.output_image_size = output_image_size
        self.data_loader_func = data_loader_func
        self.latent_vector_seed = latent_vector_seed
        self.training_seed = training_seed
        self.batch_size = batch_size
        self.sample_image_count = sample_image_count
        self.sample_image_per_row = sample_image_per_row
        self.sample_per_sample_image = sample_per_sample_image
        self.sample_per_loss_record = sample_per_loss_record
        self.sample_per_save_point = sample_per_save_point
        self.save_point_per_phase = save_point_per_phase
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_module_learning_rate = generator_module_learning_rate
        self.mapping_module_learning_rate = mapping_module_learning_rate
        self.generator_module_betas = generator_module_betas
        self.discriminator_betas = discriminator_betas
        self.mapping_module_betas = mapping_module_betas
        self.device = device

        # Array of image sizes.
        self.image_sizes = []
        size = 4
        while size <= self.output_image_size:
            self.image_sizes.append(size)
            size *= 2

        # Random fixed inputs for generating example images.
        self.latent_vector_tasks = LatentVectorFileTasks(self)
        self.noise_image_tasks = NoiseImageFileTasks(self)

        # Initial models.
        self.initial_mapping_module_tasks = InitialMappingModuleTasks(self)
        self.initial_generator_module_tastks = InitialGeneratorModuleTasks(self)
        self.initial_discriminator_tasks = InitialDiscriminatorTasks(self)
        self.initial_rng_state_tasks = InitialRngStateTasks(self)

        # The phases
        self.stabilize_phases = []
        self.transition_phases = []
        self.stabilize_phases.append(
            TrainingPhaseTasks(self, STABILIZE_PHASE_NAME, 4,
                               self.initial_mapping_module_tasks.file_name,
                               self.initial_generator_module_tastks.file_name,
                               self.initial_discriminator_tasks.file_name,
                               self.initial_rng_state_tasks.file_name))
        size = 8
        while size <= self.output_image_size:
            self.transition_phases.append(
                TrainingPhaseTasks(
                    self, TRANSITION_PHASE_NAME, size,
                    self.stabilize_phases[-1].mapping_module_tasks.file_name(self.save_point_per_phase),
                    self.stabilize_phases[-1].generator_module_tasks.file_name(self.save_point_per_phase),
                    self.stabilize_phases[-1].discriminator_tasks.file_name(self.save_point_per_phase),
                    self.stabilize_phases[-1].rng_state_tasks.file_name(self.save_point_per_phase)))
            self.stabilize_phases.append(
                TrainingPhaseTasks(
                    self, STABILIZE_PHASE_NAME, size,
                    self.transition_phases[-1].mapping_module_tasks.file_name(self.save_point_per_phase),
                    self.transition_phases[-1].generator_module_tasks.file_name(self.save_point_per_phase),
                    self.transition_phases[-1].discriminator_tasks.file_name(self.save_point_per_phase),
                    self.transition_phases[-1].rng_state_tasks.file_name(self.save_point_per_phase)))
            size *= 2

        

    def sample_latent_vectors(self, count):
        return torch.randn(count,
                           self.style_gan_spec.latent_vector_size,
                           device=self.device)

    def save_initial_models(self):
        torch.manual_seed(self.training_seed)

        mapping_module = self.style_gan_spec.mapping_module().to(self.device)
        mapping_module.initialize()
        torch_save(mapping_module, self.initial_mapping_module_tasks.file_name)

        generator_module = self.style_gan_spec.generator_module_stabilize(4).to(self.device)
        generator_module.initialize()
        torch_save(generator_module, self.initial_generator_module_tastks.file_name)

        discriminator = self.style_gan_spec.discriminator_stabilize(4).to(self.device)
        discriminator.initialize()
        torch_save(discriminator, self.initial_discriminator_tasks.file_name)

    def load_latent_vector(self):
        return torch_load(self.latent_vector_tasks.file_name)

    def load_noise_image(self) -> List[List[torch.Tensor]]:
        noise_image = []
        for i in range(len(self.image_sizes)):
            images = []
            for j in range(2):
                images.append(torch_load(self.noise_image_tasks.file_name(i, j)))
            noise_image.append(images)
        return noise_image

    def save_sample_images(self,
                           mapping_module: Module,
                           generator_module: Module,
                           batch_size: int,
                           file_name: str):
        latent_vector = self.load_latent_vector()
        noise_image = self.load_noise_image()
        self.save_sample_images_from_input_data(mapping_module,
                                                generator_module,
                                                latent_vector,
                                                noise_image,
                                                batch_size,
                                                file_name)

    def save_sample_images_from_input_data(self,
                                           mapping_module: Module,
                                           generator_module: Module,
                                           latent_vector: torch.Tensor,
                                           noise_image: List[List[torch.Tensor]],
                                           batch_size: int,
                                           file_name: str):
        mapping_module.train(False)
        generator_module.train(False)

        sample_images = None
        while (sample_images is None) or (sample_images.shape[0] < self.sample_image_count):
            if sample_images is None:
                vectors = latent_vector[:batch_size]
            else:
                limit = max(batch_size, self.sample_image_count - sample_images.shape[0])
                vectors = latent_vector[sample_images.shape[0]:sample_images.shape[0] + limit]
            mapped_vectors = mapping_module(vectors)
            images = generator_module(mapped_vectors, noise_image)
            if sample_images is None:
                sample_images = images
            else:
                sample_images = torch.cat((sample_images, images), dim=0)
        save_sample_images(sample_images.detach().cpu(), self.output_image_size, self.sample_image_per_row, file_name)
        print("Saved %s" % file_name)


class LatentVectorFileTasks(NoIndexFileTasks):
    def __init__(self, style_gan_tasks: 'StyleGanTasks'):
        super().__init__(
            style_gan_tasks.workspace,
            style_gan_tasks.prefix,
            "latent_vector",
            False)
        self.style_gan_tasks = style_gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/latent_vector.pt"

    def save_latent_vectors(self):
        torch.manual_seed(self.style_gan_tasks.latent_vector_seed)
        latent_vectors = self.style_gan_tasks.sample_latent_vectors(self.style_gan_tasks.sample_image_count)
        torch_save(latent_vectors, self.file_name)

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.save_latent_vectors())


class NoiseImageFileTasks(TwoIndicesFileTasks):
    def __init__(self, style_gan_tasks: 'StyleGanTasks'):
        super().__init__(style_gan_tasks.workspace,
                         style_gan_tasks.prefix,
                         "noise_image",
                         len(style_gan_tasks.image_sizes),
                         2,
                         False)
        self.style_gan_tasks = style_gan_tasks
        self.define_tasks()

    def file_name(self, index0: int, index1: int) -> str:
        return self.prefix + ("/noise_image_%05d_%03d.pt" % (self.style_gan_tasks.image_sizes[index0], index1))

    def sample_noise_image(self, image_size):
        return torch.randn(1, 1, image_size, image_size, device=self.style_gan_tasks.device)

    def save_noise_image(self, index0: int, index1: int):
        noise_image = self.sample_noise_image(self.style_gan_tasks.image_sizes[index0])
        fname = self.file_name(index0, index1)
        torch_save(noise_image, fname)

    def save_noise_image_func(self, index0, index1):
        return lambda: self.save_noise_image(index0, index1)

    def create_file_tasks(self, index0: int, index1: int):
        self.workspace.create_file_task(self.file_name(index0, index1), [],
                                        self.save_noise_image_func(index0, index1))


class InitialMappingModuleTasks(NoIndexFileTasks):
    def __init__(self, style_gan_tasks: StyleGanTasks):
        super().__init__(style_gan_tasks.workspace,
                         style_gan_tasks.prefix,
                         "initial_mapping_module",
                         False)
        self.style_gan_tasks = style_gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_mapping_module.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.style_gan_tasks.save_initial_models())


class InitialGeneratorModuleTasks(NoIndexFileTasks):
    def __init__(self, style_gan_tasks: StyleGanTasks):
        super().__init__(style_gan_tasks.workspace,
                         style_gan_tasks.prefix,
                         "initial_generator_module",
                         False)
        self.style_gan_tasks = style_gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_generator_module.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.style_gan_tasks.save_initial_models())


class InitialDiscriminatorTasks(NoIndexFileTasks):
    def __init__(self, style_gan_tasks: StyleGanTasks):
        super().__init__(style_gan_tasks.workspace,
                         style_gan_tasks.prefix,
                         "initial_discriminator",
                         False)
        self.style_gan_tasks = style_gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_discriminator.pt"

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.style_gan_tasks.save_initial_models())


class InitialRngStateTasks(NoIndexFileTasks):
    def __init__(self, style_gan_tasks: StyleGanTasks):
        super().__init__(style_gan_tasks.workspace,
                         style_gan_tasks.prefix,
                         "initial_rng_state",
                         False)
        self.style_gan_tasks = style_gan_tasks
        self.define_tasks()

    @property
    def file_name(self):
        return self.prefix + "/initial_rng_state.pt"

    def save_initial_rng_state(self):
        torch.manual_seed(self.style_gan_tasks.training_seed)
        save_rng_state(self.file_name)

    def create_file_task(self):
        self.workspace.create_file_task(self.file_name, [], lambda: self.save_initial_rng_state())


class TrainingPhaseTasks:
    def __init__(self,
                 style_gan_tasks: StyleGanTasks,
                 phase_name: str,
                 image_size: int,
                 previous_mapping_module_file_name: str,
                 previous_generator_module_file_name: str,
                 previous_discriminator_file_name: str,
                 previous_rng_state_file_name: str):
        self.style_gan_tasks = style_gan_tasks
        self.phase_name = phase_name
        self.image_size = image_size
        self.previous_mapping_module_file_name = previous_mapping_module_file_name
        self.previous_generator_module_file_name = previous_generator_module_file_name
        self.previous_discriminator_file_name = previous_discriminator_file_name
        self.previous_rng_state_file_name = previous_rng_state_file_name

        self.prefix = self.style_gan_tasks.prefix + "/" + self.phase_name + ("%05d" % self.image_size)

        self.save_point_count = self.style_gan_tasks.save_point_per_phase
        self.workspace = self.style_gan_tasks.workspace

        self.batch_size = self.style_gan_tasks.batch_size[self.image_size]
        self.iter_per_save_point = self.style_gan_tasks.sample_per_save_point // self.batch_size
        if self.style_gan_tasks.sample_per_save_point % self.batch_size != 0:
            self.iter_per_save_point += 1

        self.rng_state_tasks = PhaseRngStateTasks(self)
        self.mapping_module_tasks = PhaseMappingModuleTasks(self)
        self.generator_module_tasks = PhaseGeneratorModuleTasks(self)
        self.discriminator_tasks = PhaseDiscriminatorTasks(self)
        self.mapping_module_optimizer_state_tasks = PhaseMappingModuleOptimizerStateTasks(self)
        self.generator_module_optimizer_state_tasks = PhaseGeneratorModuleOptimizerStateTasks(self)
        self.discriminator_optimizer_state_tasks = PhaseDiscriminatorOptimizerStateTasks(self)
        self.generator_loss_tasks = PhaseGeneratorLossTasks(self)
        self.discriminator_loss_tasks = PhaseDiscriminatorLossTasks(self)
        self.generator_loss_plot_tasks = PhaseGeneratorLossPlotTasks(self)
        self.discriminator_loss_plot_tasks = PhaseDiscriminatorLossPlotTasks(self)

        self.rng_state_tasks.define_tasks()
        self.mapping_module_tasks.define_tasks()
        self.generator_module_tasks.define_tasks()
        self.discriminator_tasks.define_tasks()
        self.mapping_module_optimizer_state_tasks.define_tasks()
        self.generator_module_optimizer_state_tasks.define_tasks()
        self.discriminator_optimizer_state_tasks.define_tasks()
        self.generator_loss_tasks.define_tasks()
        self.discriminator_loss_tasks.define_tasks()
        self.generator_loss_plot_tasks.define_tasks()
        self.discriminator_loss_plot_tasks.define_tasks()

    def save_point_dependencies(self, save_point_index) -> List[str]:
        if save_point_index == 0:
            return [
                self.previous_rng_state_file_name,
                self.previous_mapping_module_file_name,
                self.previous_generator_module_file_name,
                self.previous_discriminator_file_name,
            ]
        else:
            return [
                self.rng_state_tasks.file_name(save_point_index - 1),
                self.mapping_module_tasks.file_name(save_point_index - 1),
                self.generator_module_tasks.file_name(save_point_index - 1),
                self.discriminator_tasks.file_name(save_point_index - 1),
                self.mapping_module_optimizer_state_tasks.file_name(save_point_index - 1),
                self.generator_module_optimizer_state_tasks.file_name(save_point_index - 1),
                self.discriminator_optimizer_state_tasks.file_name(save_point_index - 1),
                self.generator_loss_tasks.file_name(save_point_index - 1),
                self.discriminator_loss_tasks.file_name(save_point_index - 1)
            ]

    def create_mapping_module(self):
        return self.style_gan_tasks.style_gan_spec.mapping_module().to(self.style_gan_tasks.device)

    def create_generator_module(self):
        if self.phase_name == STABILIZE_PHASE_NAME:
            return self.style_gan_tasks.style_gan_spec.generator_module_stabilize(self.image_size) \
                .to(self.style_gan_tasks.device)
        else:
            return self.style_gan_tasks.style_gan_spec.generator_module_transition(self.image_size) \
                .to(self.style_gan_tasks.device)

    def create_discriminator(self):
        if self.phase_name == STABILIZE_PHASE_NAME:
            return self.style_gan_tasks.style_gan_spec.discriminator_stabilize(self.image_size) \
                .to(self.style_gan_tasks.device)
        else:
            return self.style_gan_tasks.style_gan_spec.discriminator_transition(self.image_size) \
                .to(self.style_gan_tasks.device)

    def save_save_point_zero_files(self):
        os.makedirs(self.prefix)

        load_rng_state(self.previous_rng_state_file_name)
        M = self.load_mapping_module(self.previous_mapping_module_file_name)
        G = self.load_generator_module(self.previous_generator_module_file_name)
        D = self.load_discriminator(self.previous_discriminator_file_name)
        M_optim = Adam(M.parameters(),
                       lr=self.style_gan_tasks.mapping_module_learning_rate,
                       betas=self.style_gan_tasks.mapping_module_betas)
        G_optim = Adam(G.parameters(),
                       lr=self.style_gan_tasks.generator_module_learning_rate,
                       betas=self.style_gan_tasks.generator_module_betas)
        D_optim = Adam(D.parameters(),
                       lr=self.style_gan_tasks.discriminator_learning_rate,
                       betas=self.style_gan_tasks.discriminator_betas)
        generator_loss = torch.Tensor([0])
        discriminator_loss = torch.Tensor([0])

        torch_save(M.state_dict(), self.mapping_module_tasks.file_name(0))
        torch_save(G.state_dict(), self.generator_module_tasks.file_name(0))
        torch_save(D.state_dict(), self.discriminator_loss_tasks.file_name(0))
        torch_save(M_optim.state_dict(), self.mapping_module_optimizer_state_tasks.file_name(0))
        torch_save(G_optim.state_dict(), self.generator_module_optimizer_state_tasks.file_name(0))
        torch_save(D_optim.state_dict(), self.discriminator_optimizer_state_tasks.file_name(0))
        torch_save(generator_loss, self.generator_loss_tasks.file_name(0))
        torch_save(discriminator_loss, self.discriminator_loss_tasks.file_name(0))
        save_rng_state(self.rng_state_tasks.file_name(0))

    def load_discriminator(self, discriminator_file_name):
        D = self.create_discriminator()
        D.initialize()
        D.load_state_dict(torch_load(discriminator_file_name))
        return D

    def load_generator_module(self, generator_module_file_name):
        G = self.create_generator_module()
        G.initialize()
        G.load_state_dict(torch_load(generator_module_file_name))
        return G

    def load_mapping_module(self, mapping_module_file_name):
        M = self.create_mapping_module()
        M.initialize()
        M.load_state_dict(torch_load(mapping_module_file_name))
        return M

    def load_mapping_model_optimizer(self, M: Module, mapping_module_optimizer_state_file_name: str):
        M_optim = Adam(M.parameters(),
                       lr=self.style_gan_tasks.mapping_module_learning_rate,
                       betas=self.style_gan_tasks.mapping_module_betas)
        M_optim.load_state_dict(torch_load(mapping_module_optimizer_state_file_name))
        optimizer_to_device(M_optim, self.style_gan_tasks.device)
        return M_optim

    def load_generator_model_optimizer(self, G: Module, generator_module_optimizer_state_file_name: str):
        G_optim = Adam(G.parameters(),
                       lr=self.style_gan_tasks.generator_module_learning_rate,
                       betas=self.style_gan_tasks.generator_module_betas)
        G_optim.load_state_dict(torch_load(generator_module_optimizer_state_file_name))
        optimizer_to_device(G_optim, self.style_gan_tasks.device)
        return G_optim

    def load_discriminator_optimizer(self, D: Module, discriminator_optimizer_state_file_name: str):
        D_optim = Adam(D.parameters(),
                       lr=self.style_gan_tasks.discriminator_learning_rate,
                       betas=self.style_gan_tasks.discriminator_betas)
        D_optim.load_state_dict(torch_load(discriminator_optimizer_state_file_name))
        optimizer_to_device(D_optim, self.style_gan_tasks.device)
        return D_optim

    def sample_image_file_name(self, save_point: int, index: int):
        return self.prefix + ("/sample_image_%03d_%03d" % (save_point, index))

    def get_discriminator_next_real_image_batch(self) -> torch.Tensor:
        if self.discriminator_data_loader is None:
            self.discriminator_data_loader = self.style_gan_tasks.data_loader_func(
                self.image_size, self.batch_size, self.style_gan_tasks.device)
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
            self.generator_data_loader = self.style_gan_tasks.data_loader_func(
                self.image_size, self.batch_size, self.style_gan_tasks.device)
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

        M = self.load_mapping_module(self.mapping_module_tasks.file_name(save_point - 1))
        G = self.load_generator_module(self.generator_module_tasks.file_name(save_point - 1))
        D = self.load_discriminator(self.discriminator_tasks.file_name(save_point - 1))
        M_optim = self.load_mapping_model_optimizer(
            M, self.mapping_module_optimizer_state_tasks.file_name(save_point - 1))
        G_optim = self.load_generator_model_optimizer(
            G, self.generator_module_optimizer_state_tasks.file_name(save_point - 1))
        D_optim = self.load_discriminator_optimizer(
            D, self.discriminator_optimizer_state_tasks.file_name(save_point - 1))
        generator_loss = torch_load(self.generator_loss_tasks.file_name(save_point - 1))
        discriminator_loss = torch_load(self.discriminator_loss_tasks.file_name(save_point - 1))

        self.discriminator_data_loader = None
        self.discriminator_data_loader_iter = None
        self.generator_data_loader = None
        self.generator_data_loader_iter = None
        sample_count = 0

        batch_size = self.style_gan_tasks.batch_size[self.image_size]
        sample_image_index = 0
        loss_record_index = 0
        iter_index = 0
        total_iter_index = (save_point - 1) * self.iter_per_save_point
        total_iter_count = self.iter_per_save_point * self.style_gan_tasks.save_point_per_phase
        print("=== Training %s Phase (image-size=%d, save-point=%d) ==="
              % (self.phase_name, self.image_size, save_point))
        last_time = time.time()
        alpha = 0.0

        while sample_count < self.style_gan_tasks.sample_per_save_point:
            if self.phase_name == TRANSITION_PHASE_NAME:
                alpha = total_iter_index * 1.0 / total_iter_count
                G.alpha = alpha
                D.alpha = alpha

            if sample_count / self.style_gan_tasks.sample_per_sample_image >= sample_image_index:
                self.style_gan_tasks.save_sample_images(mapping_module=M,
                                                        generator_module=G,
                                                        batch_size=self.batch_size,
                                                        file_name=self.sample_image_file_name(save_point,
                                                                                              sample_image_index))
                sample_image_index += 1

            if True:
                real_images = self.get_discriminator_next_real_image_batch()
                latent_vectors = self.style_gan_tasks.sample_latent_vectors(batch_size)
                D.train(True)
                D.zero_grad()
                G.zero_grad()
                D_loss = self.style_gan_tasks.loss_spec.discriminator_loss(lambda x: G(M(x)), D, real_images, latent_vectors)
                D_loss.backward()
                D_optim.step()

                if sample_count / self.style_gan_tasks.sample_per_loss_record >= loss_record_index:
                    discriminator_loss.append(D_loss.item())

            if True:
                real_images = self.get_generator_next_real_image_batch()
                latent_vectors = self.style_gan_tasks.sample_latent_vectors(batch_size)
                M.train(True)
                G.train(True)
                G.zero_grad()
                D.zero_grad()
                G_loss = self.style_gan_tasks.loss_spec.generator_loss(lambda x: G(M(x)), D, real_images, latent_vectors)
                G_loss.backward()
                G_optim.step()

                if sample_count / self.style_gan_tasks.sample_per_loss_record >= loss_record_index:
                    generator_loss.append(G_loss.item())

            if sample_count / self.style_gan_tasks.sample_per_loss_record >= loss_record_index:
                loss_record_index += 1
            sample_count += batch_size

            iter_index += 1
            total_iter_index += 1
            now = time.time()
            if now - last_time > 10:
                if self.phase_name == STABILIZE_PHASE_NAME:
                    print("Showed %d real images ..." % (iter_index * batch_size))
                else:
                    print("Showed %d real images (alpha=%f) ..." % (iter_index * batch_size, alpha))
                last_time = now


        torch_save(M.state_dict(), self.mapping_module_tasks.file_name(save_point))
        torch_save(G.state_dict(), self.generator_module_tasks.file_name(save_point))
        torch_save(D.state_dict(), self.discriminator_tasks.file_name(save_point))
        torch_save(M_optim.state_dict(), self.mapping_module_optimizer_state_tasks.file_name(save_point))
        torch_save(G_optim.state_dict(), self.generator_module_optimizer_state_tasks.file_name(save_point))
        torch_save(D_optim.state_dict(), self.discriminator_optimizer_state_tasks.file_name(save_point))
        torch_save(generator_loss, self.generator_loss_tasks.file_name(save_point))
        torch_save(discriminator_loss, self.discriminator_loss_tasks.file_name(save_point))
        save_rng_state(self.rng_state_tasks.file_name(save_point))


    def process_save_point(self, save_point_index):
        if save_point_index == 0:
            self.save_save_point_zero_files()
        else:
            self.train(save_point_index)


class PhaseRngStateTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="rng_state",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/rng_state_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseMappingModuleTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="mapping_module",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/mapping_module_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseGeneratorModuleTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="generator_module",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_module_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseDiscriminatorTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="discriminator",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseMappingModuleOptimizerStateTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="mapping_module_optimizer_state",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/mapping_module_optimizer_state_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseGeneratorModuleOptimizerStateTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="generator_module_optimizer_state",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_module_optimizer_state_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseDiscriminatorOptimizerStateTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="discriminator_module_optimizer_state",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_optimizer_state_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseGeneratorLossTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="generator_loss",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_loss_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


class PhaseDiscriminatorLossTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="discriminator_loss",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_loss_%03d.pt" % index)

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        self.phase_tasks.save_point_dependencies(index),
                                        lambda: self.phase_tasks.process_save_point(index))


def plot_loss(loss, title, y_label, file_name):
    plt.figure()
    plt.plot(loss)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(file_name, format='png')
    plt.close()


class PhaseGeneratorLossPlotTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="generator_loss_plot",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/generator_loss_plot_%03d.png" % index)

    def plot_loss(self, index):
        loss = torch_load(self.phase_tasks.generator_loss_tasks.file_name(index))
        title = "Generator Loss (phase=%s, image_size=%d, save_point=%d)" % (
            self.phase_tasks.phase_name, self.phase_tasks.image_size, index)
        plot_loss(loss, title, "Loss", self.file_name(index))

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        [self.phase_tasks.generator_loss_tasks.file_name(index)],
                                        lambda: self.plot_loss(index))


class PhaseDiscriminatorLossPlotTasks(OneIndexFileTasks):
    def __init__(self, phase_tasks: TrainingPhaseTasks):
        super().__init__(
            workspace=phase_tasks.workspace,
            prefix=phase_tasks.prefix,
            command_name="discriminator_loss_plot",
            count=phase_tasks.save_point_count + 1,
            define_tasks_at_creation=False)
        self.phase_tasks = phase_tasks

    def file_name(self, index):
        return self.prefix + ("/discriminator_loss_plot_%03d.png" % index)

    def plot_loss(self, index):
        loss = torch_load(self.phase_tasks.discriminator_loss_tasks.file_name(index))
        title = "Discriminator Loss (phase=%s, image_size=%d, save_point=%d)" % (
            self.phase_tasks.phase_name, self.phase_tasks.image_size, index)
        plot_loss(loss, title, "Loss", self.file_name(index))

    def create_file_tasks(self, index):
        self.workspace.create_file_task(self.file_name(index),
                                        [self.phase_tasks.discriminator_loss_tasks.file_name(index)],
                                        lambda: self.plot_loss(index))
