from torch import nn, Tensor

from .linear_model import LinearModel


class LogisticRegression(LinearModel):
    def __init__(self, input_size: int):
        """
        Initializes a Logistic Regression model with the specified input size.

        Args:
            input_size: the model's input size
        """
        super().__init__()
        
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass.

        Args:
            x: the model's input

        Returns: the model's output
        """
        return self.linear(x).flatten()
