from hana.rindou.poser.v1.poser_gan_tasks_ver2 import PoserGanValidationSpecVer2


class CustomValidationSpec(PoserGanValidationSpecVer2):
    def __init__(self, batch_size:int = 25, example_per_batch: int=500):
        super().__init__()
        self._batch_size = batch_size
        self._example_per_batch = example_per_batch

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def example_per_batch(self) -> int:
        return self._example_per_batch
