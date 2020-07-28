import torch
from torch.nn import Module, Sequential, Conv2d, Tanh, Sigmoid

from hana.rindou.nn.image_generator_bodies import bottleneck_generator_body, UNetGeneratorBody
from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class CombineTwoImagesModule(PoserGanModule):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 body_type: str = 'bottleneck',
                 has_retouch: bool = True):
        super().__init__()
        self.has_retouch = has_retouch

        if body_type == "bottleneck":
            self.combiner_xform = bottleneck_generator_body(
                image_size=image_size,
                image_dim=8 + pose_size,
                initial_dim=initial_dim,
                bottleneck_image_size=bottleneck_image_size,
                bottleneck_block_count=bottleneck_block_count,
                initialization_method=initialization_method)
        elif body_type == "unet":
            self.combiner_xform = UNetGeneratorBody(
                image_size=image_size,
                image_dim=8 + pose_size,
                initial_dim=initial_dim,
                bottleneck_image_size=bottleneck_image_size,
                bottleneck_block_count=bottleneck_block_count,
                initialization_method=initialization_method)
        else:
            raise RuntimeError("Invalid body_type: %s" % body_type)

        init = create_init_function(initialization_method)
        self.combine_alpha_mask = Sequential(
            init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
            Sigmoid())
        if has_retouch:
            self.retouch_alpha_mask = Sequential(
                init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
                Sigmoid())
            self.retouch_color_change = Sequential(
                init(Conv2d(initial_dim, 4, kernel_size=7, stride=1, padding=3, bias=False)),
                Tanh())

    def forward(self, first_image: torch.Tensor, second_image: torch.Tensor, pose: torch.Tensor):
        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.size(0), pose.size(1), first_image.size(2), first_image.size(3))

        x = torch.cat([first_image, second_image, pose], dim=1)
        combiner_xformed = self.combiner_xform(x)
        combine_alpha_mask = self.combine_alpha_mask(combiner_xformed)
        combined_image = combine_alpha_mask * first_image + (1 - combine_alpha_mask) * second_image
        if not self.has_retouch:
            return [combined_image, combine_alpha_mask]
        else:
            retouch_alpha_mask = self.retouch_alpha_mask(combiner_xformed)
            retouch_color_change = self.retouch_color_change(combiner_xformed)
            final_image = retouch_alpha_mask * combined_image + (1 - retouch_alpha_mask) * retouch_color_change
            return [final_image, combined_image, combine_alpha_mask, retouch_alpha_mask, retouch_color_change]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1], batch[2])


class CombineTwoImageModuleSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256, pose_size: int = 12,
                 initial_dim: int = 64, bottleneck_image_size: int = 32, bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 requires_optimization: bool = True,
                 body_type: str = 'bottleneck',
                 has_retouch: bool = True):
        self._image_size = image_size
        self._pose_size = pose_size
        self._initial_dim = initial_dim
        self._bottleneck_image_size = bottleneck_image_size
        self._bottleneck_block_count = bottleneck_block_count
        self._initialization_method = initialization_method
        self._requires_optimization = requires_optimization
        self._body_type = body_type
        self._has_retouch = has_retouch

    def requires_optimization(self) -> bool:
        return self._requires_optimization

    def get_module(self) -> PoserGanModule:
        return CombineTwoImagesModule(
            image_size=self._image_size,
            pose_size=self._pose_size,
            initial_dim=self._initial_dim,
            bottleneck_image_size=self._bottleneck_image_size,
            bottleneck_block_count=self._bottleneck_block_count,
            initialization_method=self._initialization_method,
            body_type=self._body_type,
            has_retouch=self._has_retouch)

if __name__ == "__main__":
    combiner = CombineTwoImagesModule(body_type='unet')
    state_dict = combiner.state_dict()
    for key in state_dict:
        #print("\"%s\"," % key)
        print(key, state_dict[key].shape)