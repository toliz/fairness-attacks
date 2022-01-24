import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_params(self):
        return tuple(self.parameters())

    def get_grads(self):
        if next(self.parameters()).grad is None:
            return None
        else:
            return torch.cat([param.grad.view(-1) for param in self.parameters()])

    def set_params(self, params):
        with torch.no_grad():
            for p1, p2 in zip(self.parameters(), params):
                p1.copy_(p2)

    @property
    def device(self):
        return next(self.parameters()).device
