from torch import nn, Tensor

from linear_model import LinearModel


class LogisticRegression(LinearModel):
    def __init__(self, input_size: int):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
