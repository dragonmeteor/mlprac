import shutil
import torch
import time
from typing import Callable, Dict, List

import matplotlib.pyplot as plt

from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from gans.gan_loss import GanLoss
from gans.pggan import LATENT_VECTOR_SIZE, PgGan, PgGanGenerator, PgGanDiscriminator, PgGanGeneratorTransition, \
    PgGanDiscriminatorTransition
from gans.util import is_power2, torch_save, torch_load, save_sample_images
from pytasuku import Workspace

# DEFAULT_BATCH_SIZE = {
#    4: 32,
#    8: 16,
#    16: 16,
#    32: 16,
#    64: 16,
#    128: 16,
#    256: 8,
#    512: 4,
#    1024: 4
# }
DEFAULT_BATCH_SIZE = {
    4: 32,
    8: 32,
    16: 32,
    32: 32,
    64: 32,
    128: 16,
    256: 16,
    512: 8,
    1024: 8
}

STABILIZE_PHASE_NAME = "stabilize"
TRANSITION_PHASE_NAME = "transition"


class PgGanTasks:
    def __init__(self,
                 workspace: Workspace,
                 dir: str,
                 output_image_size: int,
                 loss_spec: GanLoss,
                 data_loader_func: Callable[[int, int, torch.device], DataLoader],
                 latent_vector_seed=293404984,
                 training_seed=60586483,
                 batch_size: Dict[int, int] = DEFAULT_BATCH_SIZE,
                 sample_image_count=64,
                 sample_image_per_row=8,
                 sample_per_sample_image=10000,
                 sample_per_loss_record=1000,
                 sample_per_save_point=100000,
                 save_point_per_phase=6,
                 discriminator_learning_rate=1e-4,
                 generator_learning_rate=5e-4,
                 generator_betas=(0, 0.999),
                 discriminator_betas=(0, 0.999),
                 device=torch.device('cpu')):
        self.workspace = workspace
        self.dir = dir
        self.device = device

        assert output_image_size > 4
        assert is_power2(output_image_size)
        self.output_image_size = output_image_size

        self.loss_spec = loss_spec

        self.sizes = []
        s = 4
        while s <= output_image_size:
            self.sizes.append(s)
            s *= 2

        self.data_loader_func = data_loader_func

        self.latent_vector_seed = latent_vector_seed
        self.training_seed = training_seed

        self.sample_image_count = sample_image_count
        self.sample_image_per_row = sample_image_per_row

        self.batch_size = batch_size.copy()

        self.sample_per_sample_image = sample_per_sample_image
        self.sample_per_loss_record = sample_per_loss_record
        self.sample_per_save_point = sample_per_save_point
        self.save_point_per_phase = save_point_per_phase

        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.generator_betas = generator_betas
        self.discriminator_betas = discriminator_betas

        self.latent_vector_file_name = self.dir + "/latent_vectors.pt"
        self.initial_rng_state_file_name = self.dir + "/initial_rng_state.pt"
        self.initial_generator_file_name = self.dir + "/initial_generator.pt"
        self.initial_discriminator_file_name = self.dir + "/initial_discriminator.pt"

        self.final_generator_file_name = self.dir + "/final_generator.pt"
        self.final_discriminator_file_name = self.dir + "/final_discriminator.pt"

    def iter_per_save_point(self, image_size: int):
        batch_size = self.batch_size[image_size]
        result = self.sample_per_save_point // batch_size
        if self.sample_per_save_point % batch_size != 0:
            result += 1
        return result

    def phase_dir(self, phase_name: str, image_size: int) -> str:
        return self.dir + ("/%s_%05d" % (phase_name, image_size))

    def rng_state_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/rng_state_%03d.pt" % save_point)

    def discriminator_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/discriminator_%03d.pt" % save_point)

    def generator_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/generator_%03d.pt" % save_point)

    def discriminator_loss_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/discriminator_loss_%03d.pt" % save_point)

    def generator_loss_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/generator_loss_%03d.pt" % save_point)

    def discriminator_loss_plot_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/discriminator_loss_plot_%03d.png" % save_point)

    def generator_loss_plot_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/generator_loss_plot_%03d.png" % save_point)

    def discriminator_optimizer_state_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/discriminator_optimizer_state_%03d.pt" % save_point)

    def generator_optimizer_state_file_name(self, phase_name: str, image_size: int, save_point: int):
        return self.phase_dir(phase_name, image_size) + ("/generator_optimizer_state_%03d.pt" % save_point)

    def sample_latent_vectors(self, count):
        return torch.randn(count,
                           LATENT_VECTOR_SIZE,
                           device=self.device)

    def save_latent_vectors(self):
        torch.manual_seed(self.latent_vector_seed)
        latent_vectors = self.sample_latent_vectors(self.sample_image_count)
        torch_save(latent_vectors, self.latent_vector_file_name)

    def save_rng_state(self, file_name):
        rng_state = torch.get_rng_state()
        torch_save(rng_state, file_name)

    def load_rng_state(self, file_name):
        rng_state = torch_load(file_name)
        torch.set_rng_state(rng_state)

    def save_initial_model(self):
        torch.manual_seed(self.training_seed)

        gan_spec = PgGan(4, self.device)
        G = gan_spec.generator()
        G.initialize()
        torch_save(G.state_dict(), self.initial_generator_file_name)

        D = gan_spec.discriminator()
        D.initialize()
        torch_save(D.state_dict(), self.initial_discriminator_file_name)

        self.save_rng_state(self.initial_rng_state_file_name)

    def save_initial_phase_model(self,
                                 phase_name: str,
                                 image_size: int,
                                 previous_rng_state_file_name: str,
                                 previous_generator_file_name: str,
                                 previous_discriminator_file_name: str):
        self.load_rng_state(previous_rng_state_file_name)

        if phase_name == STABILIZE_PHASE_NAME:
            G = PgGanGenerator(image_size).to(self.device)
        else:
            G = PgGanGeneratorTransition(image_size).to(self.device)
        G.initialize()
        G.load_state_dict(torch_load(previous_generator_file_name), strict=False)
        G = G.to(self.device)
        torch_save(G.state_dict(), self.generator_file_name(phase_name, image_size, 0))

        if phase_name == STABILIZE_PHASE_NAME:
            D = PgGanDiscriminator(image_size).to(self.device)
        else:
            D = PgGanDiscriminatorTransition(image_size).to(self.device)
        D.initialize()
        D.load_state_dict(torch_load(previous_discriminator_file_name), strict=False)
        D = D.to(self.device)
        torch_save(D.state_dict(), self.discriminator_file_name(phase_name, image_size, 0))

        G_optim = Adam(G.parameters(), lr=self.generator_learning_rate, betas=self.generator_betas)
        torch_save(G_optim.state_dict(),
                   self.generator_optimizer_state_file_name(phase_name, image_size, 0))

        D_optim = Adam(D.parameters(), lr=self.discriminator_learning_rate, betas=self.discriminator_betas)
        torch_save(D_optim.state_dict(),
                   self.discriminator_optimizer_state_file_name(phase_name, image_size, 0))

        self.save_rng_state(self.rng_state_file_name(phase_name, image_size, 0))

    def get_next_real_image_batch(self, image_size: int, batch_size: int) -> torch.Tensor:
        if self.data_loader is None:
            self.data_loader = self.data_loader_func(image_size, batch_size, self.device)
        if self.data_loader_iter is None:
            self.data_loader_iter = self.data_loader.__iter__()
        try:
            output = self.data_loader_iter.__next__()[0]
        except StopIteration:
            self.data_loader_iter = self.data_loader.__iter__()
            output = self.data_loader_iter.__next__()[0]
        return output

    def sample_images_file_name(self, phase_name: str,
                                image_size: int,
                                save_point: int,
                                index: int) -> str:
        return self.phase_dir(phase_name, image_size) + ("/sample_images_%03d_%03d.png" % (save_point, index))

    def generate_sample_images_from_latent_vectors(self, generator: Module, latent_vectors: torch.Tensor, batch_size: int, file_name: str):
        generator.train(False)
        sample_images = None
        while (sample_images is None) or (sample_images.shape[0] < self.sample_image_count):
            if sample_images is None:
                vectors = latent_vectors[:batch_size]
            else:
                limit = max(batch_size, self.sample_image_count - sample_images.shape[0])
                vectors = latent_vectors[sample_images.shape[0]:sample_images.shape[0]+limit]
            images = generator(vectors)
            if sample_images is None:
                sample_images = images
            else:
                sample_images = torch.cat((sample_images, images), dim=0)
        save_sample_images(sample_images.detach().cpu(), self.output_image_size, self.sample_image_per_row, file_name)
        print("Saved %s" % file_name)

    def generate_sample_images(self, generator: Module, phase_name: str,
                               batch_size: int, image_size: int, save_point: int, sample_image_index: int):
        file_name = self.sample_images_file_name(phase_name, image_size, save_point, sample_image_index)
        latent_vectors = torch_load(self.latent_vector_file_name)
        self.generate_sample_images_from_latent_vectors(generator, latent_vectors, batch_size, file_name)

    def optimizer_to_device(self, optim):
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def train(self, phase_name: str, image_size: int, save_point: int):
        previous_rng_state_file_name = self.rng_state_file_name(phase_name, image_size, save_point - 1)
        previous_generator_file_name = self.generator_file_name(phase_name, image_size, save_point - 1)
        previous_discriminator_file_name = self.discriminator_file_name(phase_name, image_size, save_point - 1)
        previous_generator_optimizer_state_file_name = self.generator_optimizer_state_file_name(
            phase_name, image_size, save_point - 1)
        previous_discriminator_optimizer_state_file_name = self.discriminator_optimizer_state_file_name(
            phase_name, image_size, save_point - 1)

        self.load_rng_state(previous_rng_state_file_name)

        if phase_name == STABILIZE_PHASE_NAME:
            G = PgGanGenerator(image_size).to(self.device)
        else:
            G = PgGanGeneratorTransition(image_size).to(self.device)
        G.load_state_dict(torch_load(previous_generator_file_name))
        G = G.to(self.device)

        if phase_name == STABILIZE_PHASE_NAME:
            D = PgGanDiscriminator(image_size).to(self.device)
        else:
            D = PgGanDiscriminatorTransition(image_size).to(self.device)
        D.load_state_dict(torch_load(previous_discriminator_file_name))
        D = D.to(self.device)

        G_optim = Adam(G.parameters(), lr=self.generator_learning_rate, betas=self.generator_betas)
        G_optim.load_state_dict(torch_load(previous_generator_optimizer_state_file_name))
        self.optimizer_to_device(G_optim)

        D_optim = Adam(D.parameters(), lr=self.discriminator_learning_rate, betas=self.discriminator_betas)
        D_optim.load_state_dict(torch_load(previous_discriminator_optimizer_state_file_name))
        self.optimizer_to_device(D_optim)

        self.data_loader = None
        self.data_loader_iter = None
        sample_count = 0

        generator_loss = []
        discriminator_loss = []
        batch_size = self.batch_size[image_size]
        sample_image_index = 0
        loss_record_index = 0
        iter_index = 0
        total_iter_index = (save_point - 1) * self.iter_per_save_point(image_size)
        total_iter_count = self.iter_per_save_point(image_size) * self.save_point_per_phase
        print("=== Training %s Phase (image-size=%d, save-point=%d) ===" % (phase_name, image_size, save_point))
        last_time = time.time()
        while sample_count < self.sample_per_save_point:
            if phase_name == TRANSITION_PHASE_NAME:
                alpha = total_iter_index * 1.0 / total_iter_count
                G.alpha = alpha
                D.alpha = alpha

            if sample_count / self.sample_per_sample_image >= sample_image_index:
                self.generate_sample_images(G, phase_name,
                                            batch_size, image_size, save_point - 1, sample_image_index)
                sample_image_index += 1

            if True:
                real_images = self.get_next_real_image_batch(image_size, batch_size)
                latent_vectors = self.sample_latent_vectors(batch_size)
                D.train(True)
                D.zero_grad()
                D_loss = self.loss_spec.discriminator_loss(G, D, real_images, latent_vectors)
                D_loss.backward()
                D_optim.step()

                if sample_count / self.sample_per_loss_record >= loss_record_index:
                    discriminator_loss.append(D_loss.item())

            if True:
                latent_vectors = self.sample_latent_vectors(batch_size)
                G.train(True)
                G.zero_grad()
                G_loss = self.loss_spec.generator_loss(G, D, latent_vectors)
                G_loss.backward()
                G_optim.step()

                if sample_count / self.sample_per_loss_record >= loss_record_index:
                    generator_loss.append(G_loss.item())

            if sample_count / self.sample_per_loss_record >= loss_record_index:
                loss_record_index += 1
            sample_count += batch_size

            iter_index += 1
            total_iter_index += 1
            now = time.time()
            if now - last_time > 10:
                if phase_name == STABILIZE_PHASE_NAME:
                    print("Showed %d real images ..." % (iter_index * batch_size))
                else:
                    print("Showed %d real images (alpha=%f) ..." % (iter_index * batch_size, alpha))
                last_time = now

        torch_save(G.state_dict(), self.generator_file_name(phase_name, image_size, save_point))
        torch_save(D.state_dict(), self.discriminator_file_name(phase_name, image_size, save_point))
        self.save_rng_state(self.rng_state_file_name(phase_name, image_size, save_point))
        torch.save(torch.Tensor(generator_loss),
                   self.generator_loss_file_name(phase_name, image_size, save_point - 1))
        torch.save(torch.Tensor(discriminator_loss),
                   self.discriminator_loss_file_name(phase_name, image_size, save_point - 1))
        torch.save(G_optim.state_dict(),
                   self.generator_optimizer_state_file_name(phase_name, image_size, save_point))
        torch.save(D_optim.state_dict(),
                   self.discriminator_optimizer_state_file_name(phase_name, image_size, save_point))

    def define_savepoint_tasks(self, phase_name: str, image_size: int, save_point: int,
                               dependencies: List[str], func: Callable):
        self.workspace.create_file_task(
            name=self.rng_state_file_name(phase_name, image_size, save_point),
            dependencies=dependencies,
            func=func)
        self.workspace.create_file_task(
            name=self.generator_file_name(phase_name, image_size, save_point),
            dependencies=dependencies,
            func=func)
        self.workspace.create_file_task(
            name=self.discriminator_file_name(phase_name, image_size, save_point),
            dependencies=dependencies,
            func=func)
        self.workspace.create_file_task(
            name=self.generator_optimizer_state_file_name(phase_name, image_size, save_point),
            dependencies=dependencies,
            func=func)
        self.workspace.create_file_task(
            name=self.discriminator_optimizer_state_file_name(phase_name, image_size, save_point),
            dependencies=dependencies,
            func=func)
        if save_point > 0:
            self.workspace.create_file_task(
                name=self.generator_loss_file_name(phase_name, image_size, save_point - 1),
                dependencies=dependencies,
                func=func)
            self.workspace.create_file_task(
                name=self.discriminator_loss_file_name(phase_name, image_size, save_point - 1),
                dependencies=dependencies,
                func=func)

    def plot_loss(self, loss, title, y_label, file_name):
        plt.figure()
        plt.plot(loss)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(file_name, format='png')
        plt.close()

    def plot_generator_loss(self, phase_name: str, image_size: int, save_point: int):
        loss = None
        for i in range(save_point + 1):
            new_loss = torch_load(self.generator_loss_file_name(phase_name, image_size, i))
            if loss is None:
                loss = new_loss
            else:
                loss = torch.cat((loss, new_loss), dim=0)
        self.plot_loss(loss.numpy(),
                       "Generator Loss (image_size=%d, save_point=%d)" % (image_size, save_point),
                       "Loss",
                       self.generator_loss_plot_file_name(phase_name, image_size, save_point))

    def plot_discriminator_loss(self, phase_name: str, image_size: int, save_point: int):
        loss = None
        for i in range(save_point + 1):
            new_loss = torch_load(self.discriminator_loss_file_name(phase_name, image_size, i))
            if loss is None:
                loss = new_loss
            else:
                loss = torch.cat((loss, new_loss), dim=0)
        self.plot_loss(loss.numpy(),
                       "Discriminator Loss (image_size=%d, save_point=%d)" % (image_size, save_point),
                       "Loss",
                       self.discriminator_loss_plot_file_name(phase_name, image_size, save_point))

    def define_phase_tasks(self,
                           phase_name: str,
                           image_size: int,
                           previous_rng_state_file_name: str,
                           previous_generator_file_name: str,
                           previous_discriminator_file_name: str):
        for save_point in range(self.save_point_per_phase + 1):
            if save_point == 0:
                dependencies = [
                    self.latent_vector_file_name,
                    previous_rng_state_file_name,
                    previous_generator_file_name,
                    previous_discriminator_file_name
                ]
            else:
                dependencies = [
                    self.latent_vector_file_name,
                    self.rng_state_file_name(phase_name, image_size, save_point - 1),
                    self.generator_file_name(phase_name, image_size, save_point - 1),
                    self.discriminator_file_name(phase_name, image_size, save_point - 1),
                    self.generator_optimizer_state_file_name(phase_name, image_size, save_point - 1),
                    self.discriminator_optimizer_state_file_name(phase_name, image_size, save_point - 1),
                ]

            def train_func(gan_model: PgGanTasks, save_point):
                def train_it():
                    if save_point == 0:
                        gan_model.save_initial_phase_model(
                            phase_name,
                            image_size,
                            previous_rng_state_file_name,
                            previous_generator_file_name,
                            previous_discriminator_file_name)
                    else:
                        gan_model.train(phase_name, image_size, save_point)

                return train_it

            self.define_savepoint_tasks(phase_name,
                                        image_size,
                                        save_point, dependencies,
                                        train_func(self, save_point))

        def plot_generator_loss_func(gan_model: PgGanTasks, save_point: int):
            def plot_it():
                gan_model.plot_generator_loss(phase_name, image_size, save_point)

            return plot_it

        def plot_discriminator_loss_func(gan_model: PgGanTasks, save_point: int):
            def plot_it():
                gan_model.plot_discriminator_loss(phase_name, image_size, save_point)

            return plot_it

        for save_point in range(self.save_point_per_phase):
            self.workspace.create_file_task(
                name=self.generator_loss_plot_file_name(phase_name, image_size, save_point),
                dependencies=[self.generator_loss_file_name(phase_name, image_size, i) for i in range(save_point + 1)],
                func=plot_generator_loss_func(self, save_point))
            self.workspace.create_file_task(
                name=self.discriminator_loss_plot_file_name(phase_name, image_size, save_point),
                dependencies=[self.discriminator_loss_file_name(phase_name, image_size, i) for i in
                              range(save_point + 1)],
                func=plot_discriminator_loss_func(self, save_point))

    def define_transition_phase_tasks(self,
                                      image_size: int,
                                      previous_rng_state_file_name: str,
                                      previous_generator_file_name: str,
                                      previous_discriminator_file_name: str):
        pass

    def define_initial_model_tasks(self):
        self.workspace.create_file_task(
            name=self.latent_vector_file_name,
            dependencies=[],
            func=lambda: self.save_latent_vectors())
        self.workspace.create_file_task(
            name=self.initial_rng_state_file_name,
            dependencies=[],
            func=lambda: self.save_initial_model())
        self.workspace.create_file_task(
            name=self.initial_generator_file_name,
            dependencies=[],
            func=lambda: self.save_initial_model())
        self.workspace.create_file_task(
            name=self.initial_discriminator_file_name,
            dependencies=[],
            func=lambda: self.save_initial_model())

    def define_tasks(self):
        self.define_initial_model_tasks()

        self.define_phase_tasks(
            STABILIZE_PHASE_NAME,
            image_size=4,
            previous_rng_state_file_name=self.initial_rng_state_file_name,
            previous_generator_file_name=self.initial_generator_file_name,
            previous_discriminator_file_name=self.initial_discriminator_file_name)

        image_size = 8
        while image_size <= self.output_image_size:
            self.define_phase_tasks(
                TRANSITION_PHASE_NAME,
                image_size,
                self.rng_state_file_name(STABILIZE_PHASE_NAME, image_size // 2, self.save_point_per_phase),
                self.generator_file_name(STABILIZE_PHASE_NAME, image_size // 2, self.save_point_per_phase),
                self.discriminator_file_name(STABILIZE_PHASE_NAME, image_size // 2, self.save_point_per_phase))
            self.define_phase_tasks(
                STABILIZE_PHASE_NAME,
                image_size,
                self.rng_state_file_name(TRANSITION_PHASE_NAME, image_size, self.save_point_per_phase),
                self.generator_file_name(TRANSITION_PHASE_NAME, image_size, self.save_point_per_phase),
                self.discriminator_file_name(TRANSITION_PHASE_NAME, image_size, self.save_point_per_phase))
            image_size *= 2

        finished_generator_file_name = \
            self.generator_file_name(
                STABILIZE_PHASE_NAME,
                self.output_image_size,
                self.save_point_per_phase)
        self.workspace.create_file_task(
            self.final_generator_file_name,
            [finished_generator_file_name],
            lambda: shutil.copyfile(finished_generator_file_name, self.final_generator_file_name))

        finished_discriminator_file_name = \
            self.discriminator_file_name(
                STABILIZE_PHASE_NAME,
                self.output_image_size,
                self.save_point_per_phase)
        self.workspace.create_file_task(
            self.final_discriminator_file_name,
            [finished_discriminator_file_name],
            lambda: shutil.copyfile(finished_discriminator_file_name, self.final_discriminator_file_name))

        self.workspace.create_command_task(
            self.dir + "/train",
            [
                self.final_generator_file_name,
                self.final_discriminator_file_name
            ])

        loss_plot_files = []
        size = 4
        while size <= self.output_image_size:
            if size > 4:
                for i in range(self.save_point_per_phase):
                    loss_plot_files.append(self.generator_loss_plot_file_name(TRANSITION_PHASE_NAME, size, i))
                    loss_plot_files.append(self.discriminator_loss_plot_file_name(TRANSITION_PHASE_NAME, size, i))
            for i in range(self.save_point_per_phase):
                loss_plot_files.append(self.generator_loss_plot_file_name(STABILIZE_PHASE_NAME, size, i))
                loss_plot_files.append(self.discriminator_loss_plot_file_name(STABILIZE_PHASE_NAME, size, i))
            size *= 2

        self.workspace.create_command_task(
            self.dir + "/loss_plots",
            loss_plot_files
        )
