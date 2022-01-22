from __future__ import annotations

import torch

from torch import Tensor
from typing import Tuple, List


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: List[Tensor], Y: List[Tensor], adv_mask: List[bool]):
        super().__init__()

        self.X = torch.stack(X)
        self.Y = torch.stack(Y)
        self.adv_mask = adv_mask

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.X[idx]
        y = self.Y[idx]
        return x, y
    
    def __len__(self) -> int:
        return len(self.X)
        
    def sample(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()
        
    def get_advantaged_subset(self) -> Dataset:
        x_adv = self.X[self.adv_mask]
        y_adv = self.Y[self.adv_mask]
        return Dataset([x_adv], [y_adv], [True]*len(x_adv))
    
    def get_disadvantaged_subset(self) -> Dataset:
        x_dis = self.X[[not adv for adv in self.adv_mask]]
        y_dis = self.Y[[not adv for adv in self.adv_mask]]
        return Dataset([x_dis], [y_dis], [True]*len(x_dis))
