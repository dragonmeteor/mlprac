import torch
from torch.nn.functional import leaky_relu, tanh, relu
from torch.nn.init import xavier_normal_
from torch.nn.modules.linear import Linear

from .mnist_cgan import MnistCgan


class MnistLsCganGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z_to_a = Linear(in_features=MnistCgan.LATENT_VECTOR_SIZE, out_features=1024)
        self.y_to_b = Linear(in_features=10, out_features=1024)
        self.ab_to_c = Linear(in_features=2048, out_features=1024)
        self.c_to_d = Linear(in_features=1024, out_features=MnistCgan.IMAGE_VECTOR_SIZE)
        self.initialize()

    def forward(self, latent_vector, label):
        z = latent_vector
        y = label

        a = self.z_to_a(z)
        b = self.y_to_b(y)
        ab = relu(torch.cat([a, b], dim=1))
        c = relu(self.ab_to_c(ab))
        d = self.c_to_d(c)
        return tanh(d)

    def initialize(self):
        xavier_normal_(self.z_to_a.weight)
        xavier_normal_(self.y_to_b.weight)
        xavier_normal_(self.ab_to_c.weight)
        xavier_normal_(self.c_to_d.weight)


class MnistLsCganDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_to_a = Linear(in_features=MnistCgan.IMAGE_VECTOR_SIZE, out_features=256)
        self.y_to_b = Linear(in_features=10, out_features=256)
        self.ab_to_c = Linear(in_features=512, out_features=256)
        self.c_to_score = Linear(in_features=256, out_features=1)
        self.initialize()

    def forward(self, image, label):
        x = image
        y = label

        a = self.x_to_a(x)
        b = self.y_to_b(y)
        ab = leaky_relu(torch.cat([a, b], dim=1), negative_slope=0.01)
        c = leaky_relu(self.ab_to_c(ab), negative_slope=0.01)
        return self.c_to_score(c)

    def initialize(self):
        xavier_normal_(self.x_to_a.weight)
        xavier_normal_(self.y_to_b.weight)
        xavier_normal_(self.ab_to_c.weight)
        xavier_normal_(self.c_to_score.weight)


class MnistLsCgan(MnistCgan):
    def __init__(self, device=torch.device('cpu')):
        super().__init__(device)

    def discriminator(self):
        return MnistLsCganDiscriminator().to(self.device)

    def generator(self):
        return MnistLsCganGenerator().to(self.device)

    def discriminator_loss(self, batch_size=None):
        def loss(real_logit, fake_logit):
            real_prob = real_logit
            real_diff = real_prob - 1.0
            real_loss = real_diff.mul(real_diff).mean() / 2.0

            fake_prob = fake_logit
            fake_loss = fake_prob.mul(fake_prob).mean() / 2.0

            return real_loss + fake_loss

        return loss

    def generator_loss(self, batch_size=None):
        def loss(fake_logit):
            fake_prob = fake_logit
            fake_diff = fake_prob - 1.0
            return fake_diff.mul(fake_diff).mean() / 2.0

        return loss
