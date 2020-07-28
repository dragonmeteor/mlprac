from typing import Tuple

from hana.rindou.poser.v1.poser_gan_tasks_ver2 import PoserGanTrainingSpecVer2


class CustomTrainingSpec(PoserGanTrainingSpecVer2):
    def __init__(self,
                 save_point_count: int = 12,
                 example_per_save_point: int = 250000,
                 generator_learning_rate: float = 1e-4,
                 discriminator_learning_rate: float = 3e-4,
                 batch_size: int = 25,
                 generator_betas: Tuple[float, float] = (0.5, 0.999),
                 discriminator_betas: Tuple[float, float] = (0.5, 0.999),
                 random_seed: int = 963852741):
        super().__init__()
        self._save_point_count = save_point_count
        self._example_per_save_point = example_per_save_point
        self._generator_learning_rate = generator_learning_rate
        self._discriminator_learning_rate = discriminator_learning_rate
        self._batch_size = batch_size
        self._generator_betas = generator_betas
        self._discriminator_betas = discriminator_betas
        self._random_seed = random_seed

    @property
    def save_point_count(self) -> int:
        return self._save_point_count

    @property
    def example_per_save_point(self) -> int:
        return self._example_per_save_point

    def generator_learning_rate(self, save_point_index: int, global_example_count: int) -> float:
        return self._generator_learning_rate

    def discriminator_learning_rate(self, save_point_index: int, global_example_count: int) -> float:
        return self._discriminator_learning_rate

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def generator_betas(self) -> Tuple[float, float]:
        return self._generator_betas

    @property
    def discriminator_betas(self) -> Tuple[float, float]:
        return self._discriminator_betas

    @property
    def random_seed(self) -> int:
        return self._random_seed
