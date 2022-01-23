from __future__ import annotations

import torch

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, Union, List


class Dataset(TorchDataset):
    def __init__(self, X: Tensor, Y: Tensor, adv_mask: torch.BoolTensor):
        super().__init__()

        self.X = X
        self.Y = Y
        self.adv_mask = adv_mask

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        return self.X[idx], self.Y[idx], self.adv_mask[idx]
    
    def __len__(self) -> int:
        return len(self.X)
        
    def sample(self) -> Tuple[Tensor, Tensor]:
        rand_idx = torch.randint(high=len(self.X), size=(1,))
        return self[rand_idx]

    def get_advantaged_subset(self) -> Dataset:
        return self.X[self.adv_mask]
    
    def get_disadvantaged_subset(self) -> Dataset:
        return self.X[~self.adv_mask]


class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        X = torch.concat([d.X for d in datasets])
        Y = torch.concat([d.Y for d in datasets])
        adv_mask = torch.concat([d.adv_mask for d in datasets])

        assert isinstance(adv_mask, torch.BoolTensor)
        super().__init__(X, Y, adv_mask)
