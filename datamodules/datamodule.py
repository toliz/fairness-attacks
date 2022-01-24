import os
from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import ConcatDataset, Dataset


class Datamodule(pl.LightningDataModule, metaclass=ABCMeta):
    def __init__(self, file_name: str, data_dir: str, batch_size: int):
        super().__init__()

        self.file_name = file_name
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.path_to_file = os.path.join(self.data_dir, self.get_target_file_name())

        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        urlretrieve(
            url=f'https://raw.githubusercontent.com/Ninarehm/attack/master/Fairness_attack/data/{self.file_name}',
            filename=self.path_to_file
        )

    def setup(self, stage: Optional[str] = None) -> None:
        npz = np.load(self.path_to_file, allow_pickle=True)

        x_train, x_test = torch.tensor(npz['X_train']).float(), torch.tensor(npz['X_test']).float()
        y_train, y_test = torch.tensor(npz['Y_train']).int(), torch.tensor(npz['Y_test']).int()

        if stage in (None, 'fit'):
            self.train_data = Dataset(
                X=x_train,
                Y=y_train,
                adv_mask=self.get_advantaged_mask(x_train)
            )

        if stage in (None, 'test'):
            self.test_data = Dataset(
                X=x_test,
                Y=y_test,
                adv_mask=self.get_advantaged_mask(x_test)
            )

    def get_input_size(self) -> Tuple:
        return tuple(self.train_data[0][0].shape)

    @abstractmethod
    def get_target_file_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_sensitive_index(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_advantaged_value(self) -> object:
        raise NotImplementedError()

    def get_advantaged_mask(self, features: torch.Tensor) -> torch.BoolTensor:
        sensitive_idx = self.get_sensitive_index()
        advantaged_value = self.get_advantaged_value()

        return features[:, sensitive_idx] == advantaged_value

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, self.batch_size, shuffle=True, drop_last=True, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, len(self.test_data), num_workers=4)

    def get_train_dataset(self) -> Dataset:
        return self.train_data
    
    def get_test_dataset(self) -> Dataset:
        return self.test_data

    def update_train_dataset(self, dataset: Dataset) -> None:
        self.train_data = ConcatDataset([self.train_data, dataset])
