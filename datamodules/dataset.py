from __future__ import annotations

import torch

from torch import Tensor, BoolTensor, IntTensor
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple, Union, List


class Dataset(TorchDataset):
    def __init__(self, X: Tensor, Y: IntTensor, adv_mask: BoolTensor):
        super().__init__()

        self.X = X
        self.Y = Y
        self.adv_mask = adv_mask

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, IntTensor, BoolTensor]:
        return self.X[idx], self.Y[idx], self.adv_mask[idx]
    
    def __len__(self) -> int:
        return len(self.X)
        
    def sample(self) -> Tuple[Tensor, IntTensor, BoolTensor]:
        rand_idx = torch.randint(high=len(self.X), size=(1,))
        x, y, adv_mask = self[rand_idx]
        
        return x.squeeze(0), y.squeeze(0), adv_mask.squeeze(0)

    def get_advantaged_subset(self) -> Dataset:
        return Dataset(self.X[self.adv_mask], self.Y[self.adv_mask], self.adv_mask[self.adv_mask])
    
    def get_disadvantaged_subset(self) -> Dataset:
        return Dataset(self.X[~self.adv_mask], self.Y[~self.adv_mask], self.adv_mask[~self.adv_mask])

    def get_positive_count(self) -> int:
        return self.Y.bool().sum()

    def get_negative_count(self) -> int:
        return (~self.Y.bool()).sum()

class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        X = torch.concat([d.X for d in datasets])
        Y = torch.concat([d.Y for d in datasets])
        adv_mask = torch.concat([d.adv_mask for d in datasets])

        assert isinstance(X, Tensor)
        assert isinstance(Y, IntTensor)
        assert isinstance(adv_mask, BoolTensor)
        super().__init__(X, Y, adv_mask)
