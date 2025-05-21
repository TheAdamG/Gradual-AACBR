import torch
from deeparguing.irrelevance_edge_weights.compute_irrelevance import ComputeIrrelevance
import matplotlib.pyplot as plt


class FeatureWeightedIrrelevance(ComputeIrrelevance):

    def __init__(self, no_features):
        super(FeatureWeightedIrrelevance, self).__init__()

        # TODO: Consider other methods of initialising weights
        self.W = torch.nn.Parameter(torch.Tensor(no_features))
        torch.nn.init.normal_(self.W)

    def forward(self, new_cases: torch.Tensor, X_train: torch.tensor) -> torch.Tensor:
        # new_cases: (B, f) 
        # X_train: (n, f) 

        new_cases_score = torch.matmul(new_cases, self.W) # (B, 1)
        X_train_score = torch.matmul(X_train, self.W) # (n, 1)

        new_cases_score = new_cases_score.unsqueeze(1) # (B, 1, 1)
        X_train_score = X_train_score.unsqueeze(0) # (1, n, 1)

        result = torch.sigmoid(new_cases_score - X_train_score) # (B, n)

        return result 


    def plot_parameters(self):
        weights = self.W.detach().cpu().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(weights)), weights)
        for i, value in enumerate(weights):
            plt.text(i, value + (0.1 * (-1 if value <= 0 else 1)),
                     str(round(value, 3)), ha='center', fontsize=6)
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Feature Attribution Weights')
        plt.show()
