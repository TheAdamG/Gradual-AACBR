
import torch
from abc import abstractmethod, ABCMeta
from typing import Any

class ComputeBaseScores(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass