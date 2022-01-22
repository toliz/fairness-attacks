from torch import nn, Tensor

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        raise NotImplementedError()
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
