from torch.nn import Module, Sequential, Linear, InstanceNorm2d, ReLU

from rnn00.init_func import create_init_function
from rnn00.view_change import ViewVectorAsOneChannelImage, ViewImageAsVector


def LinearBlock(in_features: int, out_features: int, initialization_method='he') -> Module:
    init = create_init_function(initialization_method)
    return Sequential(
        init(Linear(in_features, out_features)),
        ViewVectorAsOneChannelImage(),
        InstanceNorm2d(num_features=1),
        ReLU(inplace=True),
        ViewImageAsVector())