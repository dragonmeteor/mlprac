from torch.nn import Module, Sequential, Linear, InstanceNorm2d, ReLU

from hana.rindou.nn2.init_function import create_init_function
from hana.rindou.nn2.view_change import ViewVectorAsOneChannelImage, ViewImageAsVector


def LinearBlock(in_features: int, out_features: int, initialization_method='he') -> Module:
    init = create_init_function(initialization_method)
    return Sequential(
        init(Linear(in_features, out_features)),
        ViewVectorAsOneChannelImage(),
        InstanceNorm2d(num_features=1),
        ReLU(inplace=True),
        ViewImageAsVector())