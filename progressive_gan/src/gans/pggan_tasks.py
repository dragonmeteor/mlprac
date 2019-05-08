import torch
from typing import Callable, Dict

from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader

from gans.gan_loss import GanLoss
from gans.pggan import LATENT_VECTOR_SIZE, PgGan, PgGanGenerator, PgGanDiscriminator
from gans.util import is_power2, torch_save, torch_load, save_sample_images
from pytasuku import Workspace

DEFAULT_BATCH_SIZE = {
    4: 16,
    8: 16,
    16: 16,
    32: 16,
    64: 16,
    128: 16,
    256: 8,
    512: 4,
    1024: 4
}

STABILIZE_PHASE_NAME = "stabilize"
TRANSITION_PHASE_NAME = "transition"


class PgGanTasks:
    def __init__(self,
                 workspace: Workspace,
                 dir: str,
                 output_image_size: int,
                 loss_spec: GanLoss,
                 data_loader_func: Callable[[int, int], DataLoader],
                 latent_vector_seed=293404984,
                 training_seed=60586483,
                 batch_size: Dict[int, int] = DEFAULT_BATCH_SIZE,
                 sample_image_count=64,
                 sample_image_per_row=8,
                 sample_per_sample_image=100000,
                 sample_per_loss_record=1000,
                 sample_per_save_point=100000,
                 save_point_per_phase=6,
                 learning_rate=1e-4,
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

        self.learning_rate = learning_rate
        self.generator_betas = generator_betas
        self.discriminator_betas = discriminator_betas

        self.latent_vector_file_name = self.dir + "/latent_vectors.pt"
        self.initial_rng_state_file_name = self.dir + "/initial_rng_state.pt"
        self.initial_generator_file_name = self.dir + "/initial_generator.pt"
        self.initial_discriminator_file_name = self.dir + "/initial_discriminator.pt"

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

    def sample_latent_vectors(self, count):
        return torch.rand(count,
                          LATENT_VECTOR_SIZE,
                          device=self.device) * 2.0 - 1.0

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

    def save_initial_stabilize_phase_model(self,
                                           image_size: int,
                                           previous_rng_state_file_name: str,
                                           previous_generator_file_name: str,
                                           previous_discriminator_file_name: str):
        self.load_rng_state(previous_rng_state_file_name)

        G = PgGanGenerator(image_size).to(self.device)
        G.initialize()
        G.load_state_dict(torch_load(previous_generator_file_name), strict=False)
        torch_save(G.state_dict(), self.generator_file_name(STABILIZE_PHASE_NAME, image_size, 0))

        D = PgGanDiscriminator(image_size).to(self.device)
        D.initialize()
        D.load_state_dict(torch_load(previous_discriminator_file_name), strict=False)
        torch_save(D.state_dict(), self.discriminator_file_name(STABILIZE_PHASE_NAME, image_size, 0))

        self.save_rng_state(self.rng_state_file_name(STABILIZE_PHASE_NAME, image_size, 0))

    def get_next_real_image_batch(self, image_size: int, batch_size: int) -> torch.Tensor:
        if self.data_loader is None:
            self.data_loader = self.data_loader_func(image_size, batch_size)
        if self.data_loader_iter is None:
            self.data_loader_iter = self.data_loader.__iter__()
        try:
            output = self.data_loader_iter.__next__()
        except StopIteration:
            self.data_loader_iter = self.data_loader.__iter__()
            output = self.data_loader_iter.__next__()
        return output

    def sample_images_file_name(self, phase_name: str,
                                image_size: int,
                                save_point: int,
                                index: int) -> str:
        return self.phase_dir(phase_name, image_size) + ("/sample_images_%03d_%03d.png" % (save_point, index))

    def generate_sample_images(self,
                               generator: Module,
                               batch_size: int,
                               phase_name: str,
                               image_size: int,
                               save_point: int,
                               sample_image_index: int):
        generator.train(False)
        sample_images = None
        while (sample_images is None) or (sample_images.shape[0] < self.sample_image_count):
            latent_vectors = self.sample_latent_vectors(batch_size)
            images = generator(latent_vectors)
            if sample_images is None:
                sample_images = images
            else:
                limit = max(batch_size, self.sample_image_count - sample_images.shape[0])
                sample_images = torch.cat((sample_images, images[:limit]), dim=0)
        save_sample_images(sample_images,
                           self.output_image_size,
                           self.sample_image_per_row,
                           self.sample_images_file_name(phase_name,
                                                        image_size,
                                                        save_point,
                                                        sample_image_index))

    def train_stabilize_phase(self, image_size: int, save_point: int):
        previous_rng_state_file_name = self.rng_state_file_name(STABILIZE_PHASE_NAME, image_size, save_point - 1)
        previous_generator_file_name = self.generator_file_name(STABILIZE_PHASE_NAME, image_size, save_point - 1)
        previous_discriminator_file_name = self.discriminator_file_name(STABILIZE_PHASE_NAME, image_size,
                                                                        save_point - 1)

        self.load_rng_state(previous_rng_state_file_name)
        G = PgGanGenerator(image_size).to(self.device)
        G.load_state_dict(torch_load(previous_generator_file_name))
        D = PgGanDiscriminator(image_size).to(self.device)
        D.load_state_dict(torch_load(previous_discriminator_file_name))
        G_optim = Adam(G.parameters(), lr=self.learning_rate, betas=self.generator_betas)
        D_optim = Adam(D.parameters(), lr=self.learning_rate, betas=self.discriminator_betas)

        self.data_loader = None
        self.data_loader_iter = None
        sample_count = 0

        generator_loss = []
        discriminator_loss = []
        batch_size = self.batch_size[image_size]
        sample_image_index = 0
        loss_record_index = 0
        print("=== Training Stabilize Phase (image-size=%d, save-point=%d) ===" % (image_size, save_point))
        while sample_count < self.sample_per_save_point:
            if sample_count / self.sample_per_sample_image >= sample_image_index:
                self.generate_sample_images(G,
                                            STABILIZE_PHASE_NAME,
                                            image_size,
                                            save_point - 1,
                                            sample_image_index)
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
                D.zero_grad()
                G.zero_grad()
                G_loss = self.loss_spec.generator_loss(G, D, latent_vectors)
                G_loss.backward()
                G_optim.step()

                if sample_count / self.sample_per_loss_record >= loss_record_index:
                    generator_loss.append(G_loss.item())

            if sample_count / self.sample_per_loss_record >= loss_record_index:
                loss_record_index += 1
            sample_count += batch_size

    def define_stabilize_phase_tasks(self,
                                     image_size: int,
                                     previous_rng_state_file_name: str,
                                     previous_generator_file_name: str,
                                     previous_discriminator_file_name: str):
        self.workspace.create_file_task(
            name=self.rng_state_file_name(STABILIZE_PHASE_NAME, image_size, 0),
            dependencies=[
                previous_rng_state_file_name,
                previous_generator_file_name,
                previous_discriminator_file_name
            ],
            func=lambda: self.save_initial_stabilize_phase_model(
                image_size,
                previous_rng_state_file_name,
                previous_generator_file_name,
                previous_discriminator_file_name))

    def define_transition_phase_tasks(self,
                                      image_size: int,
                                      previous_rng_state_file_name: str,
                                      previous_generator_file_name: str,
                                      previous_discriminator_file_name: str):
        pass

    def define_tasks(self):
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

        self.define_stabilize_phase_tasks(
            image_size=4,
            previous_rng_state_file_name=self.initial_rng_state_file_name,
            previous_generator_file_name=self.initial_generator_file_name,
            previous_discriminator_file_name=self.initial_discriminator_file_name)
