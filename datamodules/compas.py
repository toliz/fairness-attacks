from .datamodule import Datamodule

class CompasDatamodule(Datamodule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__('compas_data.npz', data_dir, batch_size)

    def get_dataset_name(self) -> str:
        return 'COMPAS'

    def get_target_file_name(self) -> str:
        return 'compas.npz'

    def get_sensitive_index(self) -> int:
        return 4

    def get_advantaged_value(self) -> object:
        return 2.0423824727201687
