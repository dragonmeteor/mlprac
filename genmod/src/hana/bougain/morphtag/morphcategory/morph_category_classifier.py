import torch
from torch.nn import Sequential, InstanceNorm2d, ReLU, Linear, Conv2d

from hana.rindou.nn2.conv import DownsampleBlock, Conv3
from hana.rindou.nn2.init_function import create_init_function
from hana.rindou.nn2.linear import LinearBlock
from hana.rindou.nn2.resnet_block import ResNetBlock
from hana.rindou.nn2.view_change import ViewImageAsVector
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class MorphCategoryClassifier(PoserGanModule):
    def __init__(self,
                 image_size: int = 256,
                 name_vector_size: int = 512,
                 num_panels: int = 5,
                 num_morph_categories: int = 6,
                 initialization_method: str = 'he'):
        super().__init__()
        self.num_panels = num_panels
        self.num_morph_categories = num_morph_categories
        self.name_vector_size = name_vector_size
        self.image_size = image_size

        init = create_init_function(initialization_method)

        input_channnels = 4 * 3
        modules = [
            Conv3(input_channnels, 16, initialization_method),
            InstanceNorm2d(16, affine=True),
            ReLU(inplace=True)
        ]
        num_channels = 16
        image_size = self.image_size
        while image_size > 1:
            modules.append(ResNetBlock(num_channels, initialization_method))
            num_new_channels = min(512, num_channels * 2)
            new_size = image_size // 2
            modules.append(init(Conv2d(num_channels, num_new_channels, kernel_size=4, stride=2, padding=1, bias=True)))
            modules.append(ReLU(inplace=True))
            image_size = new_size
            num_channels = num_new_channels
        modules.append(ViewImageAsVector())
        self.image_pathway = Sequential(*modules)

        self.name_panel_pathway = Sequential(
            LinearBlock(self.name_vector_size + self.num_panels, 512, initialization_method),
            LinearBlock(512, 512, initialization_method))

        self.combine = Sequential(
            LinearBlock(1024, 512, initialization_method),
            init(Linear(512, self.num_morph_categories)))

    def forward(self, rest_image, morph_image, diff_image, name_vec, panel):
        image_input = torch.cat([rest_image, morph_image, diff_image], dim=1)
        image_features = self.image_pathway(image_input)
        name_features = self.name_panel_pathway(torch.cat([name_vec, panel], dim=1))
        combined_features = torch.cat([image_features, name_features], dim=1)
        return self.combine(combined_features)

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1], batch[2], batch[3], batch[4])

    def classify(self, rest_image, morph_image, diff_image, name_vec, panel):
        score = self.forward(rest_image, morph_image, diff_image, name_vec, panel)
        return torch.argmax(score, dim=1)

    def classify_from_batch(self, batch):
        return self.classify(batch[0], batch[1], batch[2], batch[3], batch[4])


class MorphCategoryClassifierSpec(PoserGanModuleSpec):
    def __init__(self,
                 image_size: int = 256,
                 name_vector_size: int = 512,
                 num_panels: int = 5,
                 num_morph_categories: int = 6,
                 initialization_method: str = 'he'):
        self.num_panels = num_panels
        self.initialization_method = initialization_method
        self.num_morph_categories = num_morph_categories
        self.name_vector_size = name_vector_size
        self.image_size = image_size

    def requires_optimization(self) -> bool:
        return True

    def get_module(self) -> PoserGanModule:
        return MorphCategoryClassifier(
            self.image_size,
            self.name_vector_size,
            self.num_panels,
            self.num_morph_categories,
            self.initialization_method)


if __name__ == "__main__":
    device = torch.device("cuda")

    classifier = MorphCategoryClassifier(
        image_size=256,
        name_vector_size=512,
        num_morph_categories=6,
        initialization_method='he').to(device)

    rest_image = torch.zeros(8, 4, 256, 256, device=device)
    morph_image = torch.zeros(8, 4, 256, 256, device=device)
    diff_image = torch.zeros(8, 4, 256, 256, device=device)
    panel = torch.zeros(8, 5, device=device)
    name_vec = torch.zeros(8, 512, device=device)

    output = classifier.forward_from_batch([rest_image, morph_image, diff_image, name_vec, panel])
    print(output.shape)
