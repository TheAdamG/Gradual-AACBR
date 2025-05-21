import torch
from deeparguing.base_scores.compute_base_scores import ComputeBaseScores


class ConstantBaseScore(ComputeBaseScores):

    def __init__(self, constant):
        super(ConstantBaseScore, self).__init__()
        self.constant = constant


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        return torch.ones((batch_size), device=features.device)*self.constant



    def plot_parameters(self):
        print("Constant:", self.constant)

