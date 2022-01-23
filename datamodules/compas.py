from datamodule import Datamodule

class CompasDatamodule(Datamodule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__('compas.npz', data_dir, batch_size)

    def get_sensitive_index(self) -> int:
        return 4

    def get_advantaged_value(self) -> object:
        return 2.0423824727201687
