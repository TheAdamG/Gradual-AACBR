
import torch
from abc import abstractmethod, ABCMeta
from typing import Any

class ComputePartialOrder(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass



class CompareCases(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass



class Subtractor(CompareCases):
    def __init__(self, temperature=1., activation=lambda x: x):
        super(Subtractor, self).__init__()
        self.temperature = temperature
        self.activation = activation
    
    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.activation((attacker - target) / self.temperature)
