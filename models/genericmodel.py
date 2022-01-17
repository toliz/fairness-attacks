from abc import abstractmethod
import torch
import torch.nn as nn


class GenericModel(nn.Module):
    def __init__(self):
        super(GenericModel, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def get_params(self):
        return tuple(self.parameters())
    
    def get_grads(self):
        if next(self.parameters()).grad == None:
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
    