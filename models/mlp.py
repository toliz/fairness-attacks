import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, input_size: int, num_hidden: int, num_classes: int):
        super(MLP, self).__init__()

        self.num_classes = num_classes

        self.mlp = nn.Sequential(nn.Linear(input_size, num_hidden),
                                 nn.PReLU(),
                                 nn.Linear(num_hidden, num_classes))

    def forward(self, x: Tensor):
        out = self.mlp(x)
        return out