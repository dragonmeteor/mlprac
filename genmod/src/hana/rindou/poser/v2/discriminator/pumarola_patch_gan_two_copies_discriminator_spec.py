from hana.rindou.poser.v2.discriminator.pumarola_patch_gan_discriminator_spec import PumarolaPatchGanDiscriminator
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class PumarolaPatchGanTwoCopiesDiscriminator(PoserGanModule):
    def __init__(self,
                 image_dim: int = 4,
                 pose_dim: int = 3,
                 initial_dim: int = 64,
                 repeat_num=6,
                 initialization_method: str = 'he'):
        super().__init__()
        self.d0 = PumarolaPatchGanDiscriminator(image_dim, pose_dim, initial_dim, repeat_num, initialization_method)
        self.d1 = PumarolaPatchGanDiscriminator(image_dim, pose_dim, initial_dim, repeat_num, initialization_method)

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1], batch[2], batch[3])

    def forward(self, source_image, pose, target_image_0, target_image_1):
        return self.d0(source_image, pose, target_image_0) + self.d1(source_image, pose, target_image_1)


class PumarolaPatchGanTwoCopiesDiscriminatorSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_dim: int = 4,
                 pose_dim: int = 3,
                 initial_dim: int = 64,
                 repeat_num=6,
                 initialization_method: str = 'he'):
        self.initialization_method = initialization_method
        self.repeat_num = repeat_num
        self.initial_dim = initial_dim
        self.pose_dim = pose_dim
        self.image_dim = image_dim

    def requires_optimization(self) -> bool:
        return True

    def get_module(self) -> PoserGanModule:
        return PumarolaPatchGanTwoCopiesDiscriminator(
            self.image_dim,
            self.pose_dim,
            self.initial_dim,
            self.repeat_num,
            self.initialization_method)
