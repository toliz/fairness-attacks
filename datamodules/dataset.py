from __future__ import annotations

from typing import Tuple, Union, List

import torch
from torch import Tensor, BoolTensor, IntTensor
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, X: Tensor, Y: IntTensor, adv_mask: BoolTensor):
        """
        Initialize a Dataset with the provided features `X`, labels `Y` and adversarial mask `adv_mask`.

        Args:
            X: the features of the dataset
            Y: the labels of the dataset
            adv_mask: the adversarial binary mask, with True being the advantaged samples and False the disadvantaged
        """
        super().__init__()

        self.X = X
        self.Y = Y
        self.adv_mask = adv_mask

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, IntTensor, BoolTensor]:
        """
        Get a sample from this dataset at the specified index.

        Args:
            idx: the sample's index

        Returns: a sample as a tuple of (features, label, adv_mask)
        """
        return self.X[idx], self.Y[idx], self.adv_mask[idx]
    
    def __len__(self) -> int:
        """
        Get the dataset's length

        Returns: the length of this dataset
        """
        return len(self.X)
        
    def sample(self) -> Tuple[Tensor, IntTensor, BoolTensor]:
        """
        Randomly sample an element from this dataset.

        Returns: a sample as a tuple of (features, label, adv_mask)
        """
        rand_idx = torch.randint(high=len(self.X), size=(1,))
        x, y, adv_mask = self[rand_idx]
        
        return x.squeeze(0), y.squeeze(0), adv_mask.squeeze(0)

    def get_advantaged_subset(self) -> Dataset:
        """
        Get the advantaged subset of this dataset.

        Returns: the advantaged subset
        """
        return Dataset(self.X[self.adv_mask], self.Y[self.adv_mask], self.adv_mask[self.adv_mask])
    
    def get_disadvantaged_subset(self) -> Dataset:
        """
        Get the disadvantaged subset of this dataset.

        Returns: the disadvantaged subset
        """
        return Dataset(self.X[~self.adv_mask], self.Y[~self.adv_mask], self.adv_mask[~self.adv_mask])

    def get_positive_count(self) -> int:
        """
        Get the amount of samples with a positive label.

        Returns: the positive count of the dataset
        """
        return self.Y.bool().sum()

    def get_negative_count(self) -> int:
        """
        Get the amount of samples with a negative label.

        Returns: the negative count of the dataset
        """
        return (~self.Y.bool()).sum()


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        """
        Initializes a Dataset as a concatenation of n similarly shapes datasets in `datasets`.
        Args:
            datasets: the datasets to concatenate
        """
        X = torch.concat([d.X for d in datasets])
        Y = torch.concat([d.Y for d in datasets])
        adv_mask = torch.concat([d.adv_mask for d in datasets])

        assert isinstance(X, Tensor)
        assert isinstance(Y, IntTensor)
        assert isinstance(adv_mask, BoolTensor)
        super().__init__(X, Y, adv_mask)
