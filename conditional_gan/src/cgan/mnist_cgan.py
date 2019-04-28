import torch
from torch.nn.functional import relu, tanh, sigmoid
from torch.nn.init import xavier_uniform_, zeros_


class MnistGanGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z_to_a = torch.nn.Linear(in_features=100, out_features=200)
        self.a_dropout = torch.nn.Dropout()
        self.y_to_b = torch.nn.Linear(in_features=10, out_features=1000)
        self.b_dropout = torch.nn.Dropout()
        self.a_to_c = torch.nn.Linear(in_features=200, out_features=1200)
        self.c_dropout = torch.nn.Dropout()
        self.b_to_c = torch.nn.Linear(in_features=1000, out_features=1200)
        self.c_to_d = torch.nn.Linear(in_features=1200, out_features=784)

    def forward(self, latent_vector, label):
        z = latent_vector
        a = self.a_dropout(relu(self.z_to_a(z)))

        y = label
        b = self.b_dropout(relu(self.y_to_b(y)))

        c = self.c_dropout(relu(self.a_to_c(a) + self.b_to_c(b)))
        d = self.c_to_d(c)
        return tanh(d)

    def initialize(self):
        xavier_uniform_(self.z_to_a.weight)
        xavier_uniform_(self.y_to_b.weight)
        xavier_uniform_(self.a_to_c.weight)
        xavier_uniform_(self.b_to_c.weight)
        xavier_uniform_(self.c_to_d.weight)


class Maxout(torch.nn.Module):
    def __init__(self, in_features: int, pieces: int, out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features, pieces))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features, pieces))
        self.initialize()

    def forward(self, x):
        return (x * self.weight + self.bias).max(dim=2)

    def initialize(self):
        xavier_uniform_(self.weight)
        zeros_(self.bias)


class MnistCganDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_to_a = Maxout(in_features=768, pieces=5, out_features=240)
        self.y_to_b = Maxout(in_features=10, pieces=5, out_features=50)
        self.ab_to_c = Maxout(in_features=290, pieces=5, out_features=240)
        self.c_to_score = torch.nn.Linear(in_features=240, out_features=1)

    def forward(self, x, y):
        a = self.x_to_a(x)
        b = self.y_to_b(y)
        ab = torch.cat((a, b), 1)
        c = self.ab_to_c(ab)
        return self.c_to_score(c)

    def initialize(self):
        self.x_to_a.initialize()
        self.y_to_b.initialize()
        self.ab_to_c.initialize()
        xavier_uniform_(self.c_to_score.weight)


class MnistCgan:
    LATENT_VECTOR_SIZE = 100
    IMAGE_SIZE = 28
    IMAGE_VECTOR_SIZE = 784

    def __init__(self):
        pass

    def discriminator(self):
        return MnistCganDiscriminator()

    def generator(self):
        return MnistGanGenerator()

    def discriminator_loss(self):
        def loss(real_logit, fake_logit):
            real_prob = real_logit
            real_diff = real_prob - 1.0
            real_loss = real_diff.mul(real_diff).mean() / 2.0

            fake_prob = fake_logit
            fake_loss = fake_prob.mul(fake_prob).mean() / 2.0

            return real_loss + fake_loss

        return loss

    def generator_loss(self):
        def loss(fake_logit):
            fake_prob = fake_logit
            fake_diff = fake_prob - 1.0
            return fake_diff.mul(fake_diff).mean() / 2.0

        return loss

    def initialize_discriminator(self, discriminator: MnistCganDiscriminator):
        discriminator.initialize()

    def initialize_generator(self, generator: MnistGanGenerator):
        generator.initialize()
