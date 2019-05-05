import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from pytasuku import Workspace
from wgangp.gan import Gan
from wgangp.gan_loss import GanLoss


def torch_save(content, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        torch.save(content, f)


def torch_load(file_name):
    with open(file_name, 'rb') as f:
        return torch.load(f)


class MnistGanTasks:
    def __init__(self,
                 workspace: Workspace,
                 dir: str,
                 gan_spec: Gan,
                 loss_spec: GanLoss,
                 latent_vector_seed=293404984,
                 training_seed=60586483,
                 model_seed=38490553,
                 sample_image_count=64,
                 sample_image_per_row=8,
                 iter_per_sample_image=100,
                 iter_per_loss_record=10,
                 batch_size=100,
                 save_point_count=10,
                 real_image_file_name="data/mnist/training_full.pt",
                 real_image_count=60000,
                 learning_rate=1e-3,
                 generator_betas=(0.9, 0.999),
                 discriminator_betas=(0.5, 0.999),
                 discriminator_iter_per_generator_iter=5,
                 epoch_per_save_point=1):
        self.workspace = workspace
        self.dir = dir
        self.gan_spec = gan_spec
        self.loss_spec = loss_spec

        self.latent_vector_seed = latent_vector_seed
        self.training_seed = training_seed
        self.model_seed = model_seed

        self.device = self.gan_spec.device

        self.sample_image_count = sample_image_count
        self.sample_image_per_row = sample_image_per_row
        self.iter_per_sample_image = iter_per_sample_image

        self.iter_per_loss_record = iter_per_loss_record

        self.batch_size = batch_size

        self.save_point_count = save_point_count
        self.epoch_per_save_point = epoch_per_save_point

        self.real_image_file_name = real_image_file_name
        self.real_image_count = real_image_count

        self.latent_vector_file_name = self.dir + "/latent_vectors.pt"

        self.learning_rate = learning_rate
        self.generator_betas = generator_betas
        self.discriminator_betas = discriminator_betas
        self.discriminator_iter_per_generator_iter = discriminator_iter_per_generator_iter

        self.real_images = None

    def generator_file_name(self, save_point):
        return self.dir + "/generator_%03d.pt" % save_point

    def load_generator(self, epoch):
        generator = self.gan_spec.generator()
        generator.load_state_dict(torch_load(self.generator_file_name(epoch)))
        return generator

    def save_generator(self, generator, save_point):
        torch_save(generator.state_dict(), self.generator_file_name(save_point))

    def discriminator_file_name(self, save_point):
        return self.dir + "/discriminator_%03d.pt" % save_point

    def load_discriminator(self, save_point):
        discriminator = self.gan_spec.discriminator()
        discriminator.load_state_dict(torch_load(self.discriminator_file_name(save_point)))
        return discriminator

    def save_discriminator(self, discriminator, save_point):
        torch_save(discriminator.state_dict(), self.discriminator_file_name(save_point))

    def sample_images_file_name(self, save_point, index):
        return self.dir + "/sample_images_%03d_%03d.png" % (save_point, index)

    def rng_state_file_name(self, save_point):
        return self.dir + "/rng_state_%03d.pt" % save_point

    def save_rng_state(self, save_point):
        rng_state = torch.get_rng_state()
        torch_save(rng_state, self.rng_state_file_name(save_point))

    def load_rng_state(self, save_point):
        rng_state = torch_load(self.rng_state_file_name(save_point))
        torch.set_rng_state(rng_state)

    def generator_loss_file_name(self, save_point):
        return self.dir + "/generator_loss_%03d.pt" % save_point

    def save_generator_loss(self, generator_loss, save_point):
        torch_save(torch.Tensor(generator_loss), self.generator_loss_file_name(save_point))

    def load_generator_loss(self, save_point):
        return torch_load(self.generator_loss_file_name(save_point))

    def generator_loss_plot_file_name(self, save_point):
        return self.dir + "/generator_loss_plot_%03d.png" % save_point

    def plot_generator_loss(self, save_point):
        generator_loss = self.load_generator_loss(save_point).numpy()
        self.plot_loss(generator_loss,
                       "Generator Loss (Save point %03d)" % save_point,
                       "Loss",
                       self.generator_loss_plot_file_name(save_point))

    def discriminator_loss_file_name(self, save_point):
        return self.dir + "/discriminator_loss_%03d.pt" % save_point

    def save_discriminator_loss(self, discriminator_loss, save_point):
        torch_save(torch.Tensor(discriminator_loss), self.discriminator_loss_file_name(save_point))

    def load_discriminator_loss(self, save_point):
        return torch_load(self.discriminator_loss_file_name(save_point))

    def discriminator_loss_plot_file_name(self, save_point):
        return self.dir + "/dicriminator_loss_plot_%03d.png" % save_point

    def plot_discriminator_loss(self, save_point):
        discriminator_loss = self.load_discriminator_loss(save_point).numpy()
        self.plot_loss(discriminator_loss,
                       "Discriminator Loss (Save point %03d)" % save_point,
                       "Loss",
                       self.discriminator_loss_plot_file_name(save_point))

    def save_latent_vectors(self):
        torch.manual_seed(self.latent_vector_seed)
        latent_vectors = self.sample_latent_vector(self.sample_image_count)
        torch_save(latent_vectors, self.latent_vector_file_name)

    def save_initial_model(self):
        torch.manual_seed(self.model_seed)

        discriminator = self.gan_spec.discriminator()
        discriminator.initialize()
        self.save_discriminator(discriminator, 0)

        generator = self.gan_spec.generator()
        generator.initialize()
        self.save_generator(generator, 0)

        torch.manual_seed(self.training_seed)
        self.save_rng_state(0)

    def prepare_sample_images(self, images):
        numpy_images = images.detach().to(torch.device('cpu')).numpy()
        n = numpy_images.shape[0]
        num_rows = n // self.sample_image_per_row
        if n % self.sample_image_per_row != 0:
            num_rows += 1

        plt.figure(figsize=(num_rows, self.sample_image_per_row))
        gs = gridspec.GridSpec(num_rows, self.sample_image_per_row)

        for i in range(n):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(numpy_images[i, :, :], cmap='gray')

    def generate_sample_images(self, G, save_point, index):
        latent_vectors = torch_load(self.latent_vector_file_name).to(self.device)
        G.train(False)
        images = (G(latent_vectors).view(
            self.sample_image_count,
            self.gan_spec.image_size,
            self.gan_spec.image_size) + 1) * 0.5
        self.prepare_sample_images(images)
        plt.savefig(self.sample_images_file_name(save_point, index), format='png')
        plt.close()

    def load_real_images(self):
        if self.real_images is None:
            self.real_images = torch_load(self.real_image_file_name)[0].type(torch.float32).to(
                self.device) / 255.0 * 2.0 - 1.0

    def plot_loss(self, loss, title, y_label, file_name):
        plt.figure()
        plt.plot(loss)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(file_name, format='png')
        plt.close()

    def sample_latent_vector(self, size):
        return torch.rand(size,
                          self.gan_spec.latent_vector_size,
                          device=self.device) * 2.0 - 1.0

    def train(self, save_point):
        self.load_rng_state(save_point - 1)
        G = self.load_generator(save_point - 1)
        D = self.load_discriminator(save_point - 1)
        G_optim = Adam(G.parameters(), lr=self.learning_rate, betas=self.generator_betas)
        D_optim = Adam(D.parameters(), lr=self.learning_rate, betas=self.discriminator_betas)
        self.load_real_images()
        data_loader = DataLoader(TensorDataset(self.real_images),
                                 batch_size=self.batch_size,
                                 drop_last=True,
                                 shuffle=True)

        iter_count = 0
        sample_image_index = 0
        generator_loss = []
        discriminator_loss = []
        for epoch in range(self.epoch_per_save_point):
            print("=== Training save point %d, epoch %d ===" % (save_point, epoch))
            for batch in data_loader:
                if iter_count % self.iter_per_sample_image == 0:
                    self.generate_sample_images(G, save_point-1, sample_image_index)
                    sample_image_index += 1

                real_images = batch[0]
                real_images = real_images.view(self.batch_size, self.gan_spec.sample_size)
                real_images.requires_grad_(False)

                if True:
                    D.train(True)
                    D.zero_grad()
                    latent_vectors = self.sample_latent_vector(self.batch_size)
                    D_loss = self.loss_spec.discriminator_loss(G, D, real_images, latent_vectors)
                    D_loss.backward()
                    D_optim.step()

                    if iter_count % self.iter_per_loss_record == 0:
                        discriminator_loss.append(D_loss.item())

                if iter_count % self.discriminator_iter_per_generator_iter == 0:
                    G.train(True)
                    D.zero_grad()
                    G.zero_grad()
                    latent_vectors = self.sample_latent_vector(self.batch_size)
                    G_loss = self.loss_spec.generator_loss(G, D, latent_vectors)
                    G_loss.backward()
                    G_optim.step()

                if iter_count % self.iter_per_loss_record == 0:
                    G.train(True)
                    latent_vectors = self.sample_latent_vector(self.batch_size)
                    G_loss = self.loss_spec.generator_loss(G, D, latent_vectors).detach()
                    generator_loss.append(G_loss.item())

                iter_count += 1
                if iter_count % 100 == 0:
                    print("%d samples..." % (iter_count * self.batch_size))

        self.save_generator(G, save_point)
        self.save_discriminator(D, save_point)
        self.save_rng_state(save_point)
        self.save_generator_loss(generator_loss, save_point-1)
        self.save_discriminator_loss(discriminator_loss, save_point-1)


    def define_tasks(self):
        self.workspace.create_file_task(self.latent_vector_file_name, [], lambda: self.save_latent_vectors())

        self.workspace.create_file_task(self.generator_file_name(0), [], lambda: self.save_initial_model())
        self.workspace.create_file_task(self.discriminator_file_name(0), [], lambda: self.save_initial_model())
        self.workspace.create_file_task(self.rng_state_file_name(0), [], lambda: self.save_initial_model())

        sample_images_per_save_point = (self.real_image_count * self.epoch_per_save_point) \
                                       // (self.batch_size * self.iter_per_sample_image)
        sample_image_files = []
        loss_plot_files = []
        for save_point in range(1, self.save_point_count + 1):
            train_depedencies = [
                self.latent_vector_file_name,
                self.real_image_file_name,
                self.generator_file_name(save_point-1),
                self.discriminator_file_name(save_point-1),
                self.rng_state_file_name(save_point-1)
            ]

            def train_func(tasks, save_point):
                def train_it():
                    tasks.train(save_point)
                return train_it

            for i in range(sample_images_per_save_point):
                sample_images_file_name = self.sample_images_file_name(save_point-1, i)
                sample_image_files.append(sample_images_file_name)
                self.workspace.create_file_task(sample_images_file_name,
                                                train_depedencies,
                                                train_func(self, save_point))

            self.workspace.create_file_task(self.generator_file_name(save_point),
                                            train_depedencies,
                                            train_func(self, save_point))
            self.workspace.create_file_task(self.discriminator_file_name(save_point),
                                            train_depedencies,
                                            train_func(self, save_point))
            self.workspace.create_file_task(self.rng_state_file_name(save_point),
                                            train_depedencies,
                                            train_func(self, save_point))
            self.workspace.create_file_task(self.generator_loss_file_name(save_point-1),
                                            train_depedencies,
                                            train_func(self, save_point))
            self.workspace.create_file_task(self.discriminator_loss_file_name(save_point - 1),
                                            train_depedencies,
                                            train_func(self, save_point))

            def plot_generator_loss_func(tasks: MnistGanTasks, save_point: int):
                def plot_it():
                    tasks.plot_generator_loss(save_point)
                return plot_it

            self.workspace.create_file_task(self.generator_loss_plot_file_name(save_point-1),
                                            [self.generator_loss_file_name(save_point-1)],
                                            plot_generator_loss_func(self, save_point-1))
            loss_plot_files.append(self.generator_loss_plot_file_name(save_point-1))

            def plot_discriminator_loss_func(tasks: MnistGanTasks, save_point: int):
                def plot_it():
                    tasks.plot_discriminator_loss(save_point)
                return plot_it

            self.workspace.create_file_task(self.discriminator_loss_plot_file_name(save_point-1),
                                            [self.discriminator_loss_file_name(save_point-1)],
                                            plot_discriminator_loss_func(self, save_point-1))
            loss_plot_files.append(self.discriminator_loss_plot_file_name(save_point - 1))


        self.workspace.create_command_task(self.dir + "/sample_images", sample_image_files)
        self.workspace.create_command_task(self.dir + "/loss_plots", loss_plot_files)