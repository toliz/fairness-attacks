from abc import abstractmethod
import torch
import torch.nn as nn

from torch import Tensor


class GenericModel(nn.Module):
    def __init__(self):
        super(GenericModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def get_params(self):
        return torch.cat([param.view(-1) for param in self.nn.parameters()])
    
    def get_grads(self):
        if next(self.parameters()).grad == None:
            return None
        else:
            return torch.cat([param.grad.view(-1) for param in self.nn.parameters()])
    
    def set_params(self, params):
        idx = 0

        for p in self.nn.parameters():
            p.data = params[idx:idx+p.numel()].data.view(p.shape)
            idx += p.numel()

    @property
    def device(self):
        return next(self.parameters()).device
    