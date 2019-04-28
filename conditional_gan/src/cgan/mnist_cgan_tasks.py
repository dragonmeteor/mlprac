import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from cgan.mnist_cgan import MnistCgan
from pytasuku import Workspace


def torch_save(content, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        torch.save(content, f)


def torch_load(file_name):
    with open(file_name, 'rb') as f:
        return torch.load(f)


def to_onehot(label: torch.Tensor, class_count: int, device=torch.device):
    n = label.shape[0]
    label_onehot = torch.zeros(n, class_count, device=device)
    label_onehot.scatter_(dim=1, index=label.view(n, 1), value=1.0)
    return label_onehot


class MnistCganTasks:
    def __init__(self,
                 workspace: Workspace,
                 dir: str,
                 gan_spec: MnistCgan,
                 generator_input_seed=293404984,
                 training_seed=60586483,
                 model_seed=38490553,
                 sample_count_per_class=10,
                 save_point_count=10,
                 epoch_per_save_point = 1,
                 learning_rate = 1e-3,
                 real_image_file_name="data/mnist/training.pt",
                 device=torch.device('cuda')):
        self.workspace = workspace
        self.dir = dir
        self.gan_spec = gan_spec

        self.generator_input_seed = generator_input_seed
        self.training_seed = training_seed
        self.model_seed = model_seed

        self.save_point_count = save_point_count

        self.sample_count_per_class = sample_count_per_class
        self.epoch_per_save_point = epoch_per_save_point

        self.learning_rate = learning_rate

        self.device = device

        self.real_image_file_name = real_image_file_name
        self.generator_input_file_name = self.dir + "/generator_input.pt"

    def generator_file_name(self, save_point):
        return self.dir + "/generator_%03d.pt" % save_point

    def discriminator_file_name(self, save_point):
        return self.dir + "/discriminator_%03d.pt" % save_point

    def sample_images_file_name(self, save_point):
        return self.dir + "/sample_images_%03d.png" % save_point

    def save_generator_input(self):
        torch.manual_seed(self.generator_input_seed)
        n = 10 * self.sample_count_per_class
        label = torch.zeros(n, 10, dtype=torch.float)
        for i in range(10):
            start = i * self.sample_count_per_class
            end = (i + 1) * self.sample_count_per_class
            label[start:end, i] = 1.0
        generator_input = {
            'latent_vector': torch.rand(
                n,
                MnistCgan.LATENT_VECTOR_SIZE),
            'label': label
        }
        torch_save(generator_input, self.generator_input_file_name)

    def load_generator(self, epoch):
        generator = self.gan_spec.generator()
        generator.load_state_dict(torch_load(self.generator_file_name(epoch)))
        return generator

    def save_generator(self, discriminator, epoch):
        torch_save(discriminator.state_dict(), self.discriminator_file_name(epoch))

    def load_discriminator(self, epoch):
        discriminator = self.gan_spec.discriminator()
        discriminator.load_state_dict(torch_load(self.discriminator_file_name(epoch)))
        return discriminator

    def save_discriminator(self, discriminator, epoch):
        torch_save(discriminator.state_dict(), self.discriminator_file_name(epoch))

    def save_initial_model(self):
        torch.manual_seed(self.model_seed)

        discriminator = self.gan_spec.discriminator()
        self.gan_spec.initialize_discriminator(discriminator)
        self.save_discriminator(discriminator, 0)

        generator = self.gan_spec.generator()
        self.gan_spec.initialize_generator(generator)
        self.save_generator(generator, 0)

    def prepare_sample_images(self, images):
        numpy_images = images.detach().to(torch.device('cpu')).numpy()
        n = numpy_images.shape[0]
        num_rows = n // self.sample_count_per_class
        if n % self.sample_count_per_class != 0:
            raise RuntimeError("num images not divisible by sample_count_per_class")

        plt.figure(figsize=(num_rows, self.sample_count_per_class))
        gs = gridspec.GridSpec(num_rows, self.sample_count_per_class)

        for i in range(n):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(numpy_images[i, :, :], cmap='gray')

    def generate_sample_images(self, save_point):
        generator_input = torch_load(self.generator_input_file_name).to(self.device)
        generator = self.load_generator(save_point)
        generator.train(False)
        images = (generator(generator_input['laten_vector'], generator_input['label']).view(
            10 * self.sample_count_per_class,
            MnistCgan.IMAGE_SIZE,
            MnistCgan.IMAGE_SIZE) + 1)
        self.prepare_sample_images(images)
        plt.savefig(self.sample_images_file_name(save_point), format='png')

    def load_real_data(self):
        data = torch_load(self.real_image_file_name)
        image = data[0]
        label_raw = data[1].type(torch.int64).to(self.device)
        label_onehot = to_onehot(label_raw, 10, self.device)
        return {
            "image": image.type(torch.float32).to(self.device) / 255.0 * 2.0 - 1.0,
            "label": label_onehot
        }

    def train(self):
        pass

    def define_tasks(self):
        self.workspace.create_file_task(self.generator_input_file_name,
                                        [],
                                        lambda: self.save_generator_input())

        self.workspace.create_file_task(self.generator_file_name(0),
                                        [],
                                        lambda: self.save_initial_model())
        self.workspace.create_file_task(self.discriminator_file_name(0),
                                        [],
                                        lambda: self.save_initial_model())

        train_dependencies = [
            self.real_image_file_name,
            self.generator_file_name(0),
            self.discriminator_file_name(0)
        ]
        for save_point in range(1, self.save_point_count+1):
            self.workspace.create_file_task(self.generator_file_name(save_point),
                                            train_dependencies,
                                            lambda: self.train())
            self.workspace.create_file_task(self.discriminator_file_name(save_point),
                                            train_dependencies,
                                            lambda: self.train())

        for save_point in range(self.save_point_count + 1):
            def make_func(i):
                def gen():
                    self.generate_sample_images(i)

                return gen

            self.workspace.create_file_task(self.sample_images_file_name(save_point),
                                            [self.generator_input_file_name, self.generator_file_name(save_point)],
                                            make_func(save_point))

        self.workspace.create_command_task(self.dir + "/sample_images",
                                           [self.sample_images_file_name(i) for i in range(self.save_point_count + 1)])

if __name__ == "__main__":
    workspace = Workspace()
    gan_spec = MnistCgan()
    mnist_cgan_tasks = MnistCganTasks(workspace, "data/mnist_cgan", gan_spec)
    mnist_cgan_tasks.load_real_data()
