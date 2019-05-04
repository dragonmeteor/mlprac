import os
import torch

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from pytasuku import Workspace
from wgangp.mnist_gan import MnistGan
from wgangp.mnist_wgangp import MnistWganGp


def torch_save(content, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        torch.save(content, f)


def torch_load(file_name):
    with open(file_name, 'rb') as f:
        return torch.load(f)


class MnistWganGpTasks:
    def __init__(self,
                 workspace: Workspace,
                 dir: str,
                 gan_spec: MnistWganGp,
                 latent_vector_seed=293404984,
                 training_seed=60586483,
                 model_seed=38490553,
                 sample_image_count=64,
                 sample_image_per_row=8,
                 batch_size=MnistGan.DEFAULT_BATCH_SIZE,
                 save_point_count=10,
                 real_image_file_name="data/mnist/training.pt",
                 learning_rate=1e-3,
                 epoch_per_save_point=1):
        self.workspace = workspace
        self.dir = dir
        self.gan_spec = gan_spec

        self.latent_vector_seed = latent_vector_seed
        self.training_seed = training_seed
        self.model_seed = model_seed

        self.device = self.gan_spec.device

        self.sample_image_count = sample_image_count
        self.sample_image_per_row = sample_image_per_row

        self.batch_size = batch_size

        self.save_point_count = save_point_count
        self.epoch_per_save_point = epoch_per_save_point

        self.real_image_file_name = real_image_file_name

        self.latent_vector_file_name = self.dir + "/latent_vectors.pt"

        self.learning_rate = learning_rate

    def generator_file_name(self, save_point):
        return self.dir + "/generator_%03d.pt" % save_point

    def discriminator_file_name(self, save_point):
        return self.dir + "/discriminator_%03d.pt" % save_point

    def sample_images_file_name(self, save_point):
        return self.dir + "/sample_images_%03d.png" % save_point

    def save_latent_vectors(self):
        torch.manual_seed(self.latent_vector_seed)
        latent_vectors = torch.rand(self.sample_image_count, self.gan_spec.latent_vector_size,
                                    device=self.device) * 2.0 - 1.0
        torch_save(latent_vectors, self.latent_vector_file_name)

    def load_generator(self, epoch):
        generator = self.gan_spec.generator()
        generator.load_state_dict(torch_load(self.generator_file_name(epoch)))
        return generator

    def save_generator(self, generator, epoch):
        torch_save(generator.state_dict(), self.generator_file_name(epoch))

    def load_discriminator(self, epoch):
        discriminator = self.gan_spec.discriminator()
        discriminator.load_state_dict(torch_load(self.discriminator_file_name(epoch)))
        return discriminator

    def save_discriminator(self, discriminator, epoch):
        torch_save(discriminator.state_dict(), self.discriminator_file_name(epoch))

    def save_initial_model(self):
        torch.manual_seed(self.model_seed)

        discriminator = self.gan_spec.discriminator()
        discriminator.initialize()
        self.save_discriminator(discriminator, 0)

        generator = self.gan_spec.generator()
        self.gan_spec.initialize_generator(generator)
        self.save_generator(generator, 0)

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

    def generate_sample_images(self, save_point):
        latent_vectors = torch_load(self.latent_vector_file_name).to(self.device)
        generator = self.load_generator(save_point)
        generator.train(False)
        images = (generator(latent_vectors).view(
            self.sample_image_count,
            MnistGan.IMAGE_SIZE,
            MnistGan.IMAGE_SIZE) + 1) * 0.5
        self.prepare_sample_images(images)
        plt.savefig(self.sample_images_file_name(save_point), format='png')

    def load_real_mnist_images(self):
        real_images = torch_load(self.real_image_file_name)[0].type(torch.float32).to(
            self.device) / 255.0 * 2.0 - 1.0
        return real_images

    def train(self):
        torch.manual_seed(self.training_seed)

        G = self.load_generator(0)
        D = self.load_discriminator(0)
        G_optim = Adam(G.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        D_optim = Adam(D.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        data_loader = DataLoader(TensorDataset(self.load_real_mnist_images()), batch_size=self.batch_size, shuffle=True)

        for save_point in range(1, self.save_point_count + 1):
            for epoch in range(self.epoch_per_save_point):
                print("=== Training save point %d, epoch %d ===" % (save_point, epoch))
                batch_count = 0
                for batch in data_loader:
                    real_images = batch[0]
                    if real_images.shape[0] != self.batch_size:
                        continue
                    real_images = real_images.view(self.batch_size, MnistGan.IMAGE_VECTOR_SIZE)
                    real_images.requires_grad_(False)

                    if True:
                        D.zero_grad()
                        latent_vectors = torch.rand(self.batch_size, MnistGan.LATENT_VECTOR_SIZE,
                                                    device=self.device) * 2.0 - 1.0
                        D_loss = self.gan_spec.discriminator_loss(G, D, real_images, latent_vectors)
                        D_loss.backward()
                        D_optim.step()

                    if True:
                        D.zero_grad()
                        G.zero_grad()
                        latent_vectors = torch.rand(self.batch_size, MnistGan.LATENT_VECTOR_SIZE,
                                                    device=self.device) * 2.0 - 1.0
                        G_loss = self.gan_spec.generator_loss(G, D, latent_vectors)
                        G_loss.backward()
                        G_optim.step()

                        batch_count += 1
                        if batch_count % 100 == 0:
                            print("%d samples..." % (batch_count * self.batch_size))

            self.save_generator(G, save_point)
            self.save_discriminator(D, save_point)
            self.generate_sample_images(save_point)

    def define_tasks(self):
        self.workspace.create_file_task(self.latent_vector_file_name, [], lambda: self.save_latent_vectors())

        self.workspace.create_file_task(self.generator_file_name(0), [], lambda: self.save_initial_model())
        self.workspace.create_file_task(self.discriminator_file_name(0), [], lambda: self.save_initial_model())

        train_depedencies = [
            self.real_image_file_name,
            self.generator_file_name(0),
            self.discriminator_file_name(0)
        ]
        for save_point in range(1, self.save_point_count + 1):
            self.workspace.create_file_task(self.generator_file_name(save_point), train_depedencies, lambda: self.train())
            self.workspace.create_file_task(self.discriminator_file_name(save_point), train_depedencies,
                                            lambda: self.train())

        for save_point in range(self.save_point_count + 1):
            def make_func(i):
                def gen():
                    self.generate_sample_images(i)

                return gen

            self.workspace.create_file_task(self.sample_images_file_name(save_point),
                                            [self.latent_vector_file_name, self.generator_file_name(save_point)],
                                            make_func(save_point))

        self.workspace.create_command_task(self.dir + "/sample_images",
                                           [self.sample_images_file_name(i) for i in range(self.save_point_count + 1)])