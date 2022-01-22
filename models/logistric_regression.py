from torch import nn, Tensor

from linear_model import LinearModel


class LogisticRegression(LinearModel):
    def __init__(self):
        super().__init__()
        
        raise NotImplementedError()
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
