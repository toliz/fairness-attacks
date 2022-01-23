from .datamodule import Datamodule

class GermanCreditDatamodule(Datamodule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__('data.npz', data_dir, batch_size)

    def get_target_file_name(self) -> str:
        return 'german_credit.npz'

    def get_sensitive_index(self) -> int:
        return 36

    def get_advantaged_value(self) -> object:
        return 1.4919136877222166
