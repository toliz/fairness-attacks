from __future__ import annotations

import torch

from torch import Tensor
from typing import Tuple, List


class Dataset(torch.utils.Dataset):
    def __init__(self, X: List[Tensor], Y: List[Tensor], adv_mask: List[Tensor]):
        super().__init__()
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        raise NotImplementedError()
        
    def sample(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()
        
    def get_advantaged_subset(self) -> Dataset:
        raise NotImplementedError()
    
    def get_disadvantaged_subset(self) -> Dataset:
        raise NotImplementedError()
