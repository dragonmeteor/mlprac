from hana.bougain.landmarks.generface.face_decomposer_and_morpher import FaceDecomposerAndMorpher
from hana.bougain.landmarks.generface.keypoint.siarohin_keypoint_detector import SiarohinKeypointDetector
from hana.rindou.poser.v2.poser_gan_module import PoserGanModule
from hana.rindou.poser.v2.poser_gan_module_spec import PoserGanModuleSpec


class KeypointBasedFaceMorpher00(PoserGanModule):
    def __init__(self,
                 num_keypoints: int = 32,
                 in_channels: int = 3,
                 temperature: float = 0.1,
                 keypoint_detector_in_channels: int = 32,
                 decomposer_in_channels: int = 32,
                 num_blocks: int = 5,
                 max_channels: int = 1024,
                 keypoint_variance: float = 0.01,
                 activation: str = 'relu'):
        super().__init__()
        self.keypoint_detector = SiarohinKeypointDetector(
            num_keypoints=num_keypoints,
            in_channels=in_channels,
            temperature=temperature,
            hidden_channels=keypoint_detector_in_channels,
            num_blocks=num_blocks,
            max_channels=max_channels,
            activation=activation)
        self.face_morpher = FaceDecomposerAndMorpher(
            in_channels=in_channels,
            decomposer_in_channels=decomposer_in_channels,
            num_keypoints=num_keypoints,
            num_blocks=num_blocks,
            max_channels=max_channels,
            keypoint_variance=keypoint_variance,
            activation=activation)

    def forward(self, source_image, target_image):
        keypoints = self.keypoint_detector(target_image)[0]
        face_morpher_outputs = self.face_morpher(source_image, keypoints)
        return face_morpher_outputs + [keypoints]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])


class KeypointBasedFaceMorpherSpec00(PoserGanModuleSpec):
    def __init__(self,
                 num_keypoints: int = 32,
                 in_channels: int = 3,
                 temperature: float = 0.1,
                 keypoint_detector_in_channels: int = 32,
                 decomposer_in_channels: int = 32,
                 num_blocks: int = 5,
                 max_channels: int = 1024,
                 keypoint_variance: float = 0.01,
                 activation: str = 'relu'):
        self.num_blocks = num_blocks
        self.max_channels = max_channels
        self.keypoint_variance = keypoint_variance
        self.activation = activation
        self.decomposer_in_channels = decomposer_in_channels
        self.keypoint_detector_in_channels = keypoint_detector_in_channels
        self.temperature = temperature
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

    def requires_optimization(self) -> bool:
        return True

    def get_module(self) -> PoserGanModule:
        return KeypointBasedFaceMorpher00(
            num_keypoints=self.num_keypoints,
            in_channels=self.in_channels,
            temperature=self.temperature,
            keypoint_detector_in_channels=self.keypoint_detector_in_channels,
            decomposer_in_channels=self.decomposer_in_channels,
            num_blocks=self.num_blocks,
            max_channels=self.max_channels,
            keypoint_variance=self.keypoint_variance,
            activation=self.activation)
