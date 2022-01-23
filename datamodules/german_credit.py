from .datamodule import Datamodule

class GermanCreditDatamodule(Datamodule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__('data.npz', data_dir, batch_size)

    def get_sensitive_index(self) -> int:
        return 36

    def get_advantaged_value(self) -> object:
        return -0.6702800625998365
