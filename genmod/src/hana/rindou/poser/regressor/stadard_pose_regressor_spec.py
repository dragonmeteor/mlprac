import torch

from torch.nn import Module, Conv2d

from hana.rindou.nn.image_discriminator_body import ImageDiscriminatorBody
from hana.rindou.nn.init_function import create_init_function
from hana.rindou.poser.regressor.pose_regressor_spec import PoseRegressorSpec


class StandardPoseRegressorSpec(PoseRegressorSpec):
    def __init__(self,
                 image_size=256,
                 pose_size=6,
                 bone_parameter_count=3,
                 initialization_method='he',
                 insert_residual_blocks=False):
        super().__init__()
        self._image_size = image_size
        self._pose_size = pose_size
        self._bone_parameter_count = bone_parameter_count
        self.initialization_method = initialization_method
        self.insert_residual_blocks = insert_residual_blocks

    def image_size(self) -> int:
        return self._image_size

    def pose_size(self) -> int:
        return self._pose_size

    def bone_parameter_count(self) -> int:
        return self._bone_parameter_count

    def regressor(self) -> Module:
        return StandardPoseRegressor(
            self._image_size,
            self._pose_size,
            self.initialization_method,
            self.insert_residual_blocks)


class StandardPoseRegressor(Module):
    def __init__(self,
                 image_size=256,
                 pose_size=6,
                 initialization_method='he',
                 insert_residual_blocks=False):
        super().__init__()
        self.image_size = image_size
        self.pose_size = pose_size
        self.main = ImageDiscriminatorBody(
            image_size,
            initialization_method=initialization_method,
            insert_residual_blocks=insert_residual_blocks)
        init = create_init_function(initialization_method)
        self.pose = init(Conv2d(self.main.out_dim, pose_size, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, rest_image, posed_image):
        x = torch.cat([rest_image, posed_image], dim=1)
        return self.pose(self.main(x)).squeeze()


if __name__ == "__main__":
    cuda = torch.device("cuda")
    spec = StandardPoseRegressorSpec()
    R = spec.regressor().to(cuda)

    rest_image = torch.zeros(8, 4, 256, 256, device=cuda)
    posed_image = torch.zeros(8, 4, 256, 256, device=cuda)
    inferred_pose = R(rest_image, posed_image)
    print(inferred_pose.shape)
