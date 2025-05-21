
import torch
from abc import abstractmethod, ABCMeta
from typing import Any

class ComputeIrrelevance(torch.nn.Module, metaclass=ABCMeta):


    @abstractmethod
    def forward(self, new_cases: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def plot_parameters(self):
        pass