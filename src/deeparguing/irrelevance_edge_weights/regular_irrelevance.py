import torch
from deeparguing.irrelevance_edge_weights.compute_irrelevance import ComputeIrrelevance
import matplotlib.pyplot as plt
from deeparguing.casebase_edge_weights.compute_partial_order import ComputePartialOrder

class RegularIrrelevance(ComputeIrrelevance):

    def __init__(self, compute_partial_order: ComputePartialOrder):
        super(RegularIrrelevance, self).__init__()

        self.compute_partial_order = compute_partial_order


    def forward(self, new_cases: torch.Tensor, X_train: torch.tensor) -> torch.Tensor:
        
        new_cases = new_cases.unsqueeze(1) # (B, 1, no_features)
        X_train = X_train.unsqueeze(0) # (1, n, no_features)

        return 1 - self.compute_partial_order(new_cases, X_train)


    def plot_parameters(self):
       self.compute_partial_order.plot_parameters() 