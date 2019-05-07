import torch
from typing import Callable, Dict

from torch.utils.data import DataLoader

from gans.gan_loss import GanLoss
from gans.pggan import LATENT_VECTOR_SIZE, PgGan
from gans.util import is_power2, torch_save, torch_load
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


class PgGanTasks:
    def __init__(self,
                 workspace: Workspace,
                 dir: str,
                 image_size: int,
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

        assert image_size > 4
        assert is_power2(image_size)
        self.image_size = image_size

        self.loss_spec = loss_spec

        self.sizes = []
        s = 4
        while s <= image_size:
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

    def phase_dir(self, phase_name: str, size: int) -> str:
        return self.dir + ("/%s_%05d" % (phase_name, size))

    def rng_state_file_name(self, phase_name: str, size: int, save_point: int):
        return self.phase_dir(phase_name, size) + ("/rng_state_%03d.pt" % save_point)

    def discriminator_file_name(self, phase_name: str, size: int, save_point: int):
        return self.phase_dir(phase_name, size) + ("/discriminator_%03d.pt" % save_point)

    def generator_file_name(self, phase_name: str, size: int, save_point: int):
        return self.phase_dir(phase_name, size) + ("/generator_%03d.pt" % save_point)

    def discriminator_loss_file_name(self, phase_name: str, size: int, save_point: int):
        return self.phase_dir(phase_name, size) + ("/discriminator_loss_%03d.pt" % save_point)

    def generator_loss_file_name(self, phase_name: str, size: int, save_point: int):
        return self.phase_dir(phase_name, size) + ("/generator_loss_%03d.pt" % save_point)

    def discriminator_loss_plot_file_name(self, phase_name: str, size: int, save_point: int):
        return self.phase_dir(phase_name, size) + ("/discriminator_loss_plot_%03d.png" % save_point)

    def generator_loss_plot_file_name(self, phase_name: str, size: int, save_point: int):
        return self.phase_dir(phase_name, size) + ("/generator_loss_plot_%03d.png" % save_point)

    def sample_latent_vectors(self, size):
        return torch.rand(size,
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


    def define_stabilize_phase_tasks(self,
                                     size: int,
                                     zeroth_rng_state_file_name: str):
        pass

    def define_transition_phase_tasks(self,
                                      size: int,
                                      zeroth_rng_state_file_name: str):
        pass

    def define_tasks(self):
        self.workspace.create_file_task(self.latent_vector_file_name, [],
                                        lambda: self.save_latent_vectors())
        self.workspace.create_file_task(self.initial_rng_state_file_name, [],
                                        lambda: self.save_initial_model())
        self.workspace.create_file_task(self.initial_generator_file_name, [],
                                        lambda: self.save_initial_model())
        self.workspace.create_file_task(self.initial_discriminator_file_name, [],
                                        lambda: self.save_initial_model())

        self.define_stabilize_phase_tasks(4, self.initial_rng_state_file_name)
