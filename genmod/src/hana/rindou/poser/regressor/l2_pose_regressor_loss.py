from torch import Tensor
from torch.nn import Module
from typing import List

from hana.rindou.poser.regressor.pose_regressor_loss import PoseRegressorLoss


class L2PoseRegressorLoss(PoseRegressorLoss):
    def compute(self, R: Module, batch: List[Tensor]) -> Tensor:
        rest_image = batch[0]
        posed_image = batch[1]
        pose = batch[2]
        inferred_pose = R(rest_image, posed_image)
        return ((inferred_pose - pose) ** 2).mean()
