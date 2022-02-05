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
        """
        Initialize a Datamodule, by either downloading a dataset from a URL, or loading it from a local file,
        at the specified data directory, with its DataLoaders having a batch size of `batch_size`.

        Args:
            file_name: the dataset's npz file name from the original authors' repository
            data_dir: the directory where the dataset should be saved at (and read from)
            batch_size: the batch size for this LightningDataModule's DataLoaders
        """
        super().__init__()

        # Save params
        self.file_name = file_name
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Path to save/read the file is the data dir and the dataset's unique file name
        self.path_to_file = os.path.join(self.data_dir, self.get_target_file_name())

        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        """
        Prepares the data by downloading them if they do not exist locally
        """
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        urlretrieve(
            url=f'https://raw.githubusercontent.com/Ninarehm/attack/master/Fairness_attack/data/{self.file_name}',
            filename=self.path_to_file
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the Datamodule by loading the datasets into memory.

        Args:
            stage: the stage during which this Datamodule is being setup (e.g. 'fit', 'test', 'None')
        """
        # Load the NumPy array from the specified npz file
        npz = np.load(self.path_to_file, allow_pickle=True)

        # Extract X_train, X_test, Y_train, Y_test from the NumPy array
        x_train, x_test = torch.tensor(npz['X_train']).float(), torch.tensor(npz['X_test']).float()
        y_train, y_test = torch.tensor(npz['Y_train']).int(), torch.tensor(npz['Y_test']).int()

        if stage in (None, 'fit'):
            # If we are in the training stage (or no stage), load the train dataset into memory
            self.train_data = Dataset(
                X=x_train,
                Y=y_train,
                adv_mask=self.get_advantaged_mask(x_train)
            )

        if stage in (None, 'test'):
            # If we are in the testing stage (or no stage), load the test dataset into memory
            self.test_data = Dataset(
                X=x_test,
                Y=y_test,
                adv_mask=self.get_advantaged_mask(x_test)
            )

    def get_input_size(self) -> Tuple:
        """
        Get the individual samples' size as a tuple.

        Returns: the dataset's samples' size
        """
        return tuple(self.train_data[0][0].shape)
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        Get the Datamodule's dataset name.

        Returns: the dataset's name
        """
        raise NotImplementedError()

    @abstractmethod
    def get_target_file_name(self) -> str:
        """
        Get the file name where the dataset is stored.

        Returns: the dataset's file name
        """
        raise NotImplementedError()

    @abstractmethod
    def get_sensitive_index(self) -> int:
        """
        Get the index of the sensitive feature for this dataset's points.

        Returns: the sensitive feature index
        """
        raise NotImplementedError()

    @abstractmethod
    def get_advantaged_value(self) -> object:
        """
        Get the sensitive feature's value that corresponds to the advantaged group.

        Returns: the advantaged group's value
        """
        raise NotImplementedError()

    def get_advantaged_mask(self, features: torch.Tensor) -> torch.BoolTensor:
        """
        Get this dataset's advantaged mask, with the True values representing the points that
        are in the advantaged class, and False values the points that are in the disadvantaged.

        Args:
            features: the features (X) of the dataset

        Returns: the dataset's advantaged binary mask
        """
        sensitive_idx = self.get_sensitive_index()
        advantaged_value = self.get_advantaged_value()

        return features[:, sensitive_idx] == advantaged_value

    def train_dataloader(self) -> DataLoader:
        """
        Get the Datamodule's train DataLoader.

        Returns: the train dataloader
        """
        return DataLoader(self.train_data, self.batch_size, shuffle=True, drop_last=True, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        """
        Get the Datamodule's test DataLoader.

        Returns: the test dataloader
        """
        return DataLoader(self.test_data, self.batch_size, num_workers=4)

    def get_train_dataset(self) -> Dataset:
        """
        Get the Datamodule's corresponding train dataset.

        Returns: the train dataset
        """
        return self.train_data
    
    def get_test_dataset(self) -> Dataset:
        """
        Get the Datamodule's corresponding test dataset.

        Returns: the test dataset
        """
        return self.test_data

    def update_train_dataset(self, extra_dataset: Dataset) -> None:
        """
        Updates the train dataset by appending `extra_dataset` to it.

        Args:
            extra_dataset: the extra dataset to append to the train data
        """
        self.train_data = ConcatDataset([self.train_data, extra_dataset])
