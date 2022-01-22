import pytorch_lightning as pl

from .dataset import Dataset


class Datamodule(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        
        raise NotImplementedError()

    def get_train_dataset(self) -> Dataset:
        raise NotImplementedError()
    
    def get_test_dataset(self) -> Dataset:
        raise NotImplementedError()
