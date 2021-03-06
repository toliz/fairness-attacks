from .datamodule import Datamodule

class DrugConsumptionDatamodule(Datamodule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__('drug2_data.npz', data_dir, batch_size)

    def get_dataset_name(self) -> str:
        return 'Drug Consumption'

    def get_target_file_name(self) -> str:
        return 'drug_consumption.npz'

    def get_sensitive_index(self) -> int:
        return 12

    def get_advantaged_value(self) -> object:
        return 1.0005306447706963
