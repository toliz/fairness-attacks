from typing import Tuple, Optional

import torch
from torch import nn, Tensor


class LinearModel(nn.Module):
    def __init__(self):
        """
        Initializes a Linear Model.
        """
        super().__init__()

    def get_params(self) -> Tuple[nn.Parameter]:
        """
        Get the model's parameters.

        Returns: the model's parameters as a tuple
        """
        return tuple(self.parameters())

    def get_grads(self) -> Optional[Tensor]:
        """
        Get the model's gradients.

        Returns: the model's gradients as a concatenated tensor
        """
        if next(self.parameters()).grad is None:
            return None
        else:
            return torch.cat([param.grad.view(-1) for param in self.parameters()])

    def set_params(self, params: Tuple[nn.Parameter]):
        """
        Set the model's parameters.

        Args:
            params: the parameters to set
        """
        with torch.no_grad():
            for p1, p2 in zip(self.parameters(), params):
                p1.copy_(p2)

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model is loaded.

        Returns: the model's device
        """
        return next(self.parameters()).device
