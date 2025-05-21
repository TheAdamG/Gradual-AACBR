import torch
from deeparguing.casebase_edge_weights.compute_partial_order import ComputePartialOrder
import matplotlib.pyplot as plt


class SMPartialOrder(ComputePartialOrder):

    def __init__(self, no_features, sharpness = 1, activation = torch.sigmoid):
        super(SMPartialOrder, self).__init__()

        # TODO: Consider other methods of initialising weights
        self.W = torch.nn.Parameter(torch.Tensor(no_features))
        torch.nn.init.normal_(self.W)
        self.sharpness = sharpness
        self.activation = activation

    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if target.ndim == 1:
            target = target.unsqueeze(0)

        if attacker.ndim == 1:
            attacker = attacker.unsqueeze(0)


        attacker_threshold = torch.relu((torch.abs(attacker) - torch.abs(self.W)) * self.sharpness)
        target_threshold = torch.relu((torch.abs(target) - torch.abs(self.W)) * self.sharpness)


        return self.activation(torch.sum(attacker_threshold - target_threshold, dim=-1))




        # return torch.sigmoid((attacker_score - target_score) * self.sharpness) 


    def plot_parameters(self):
        weights = self.W.detach().cpu().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(weights)), weights)
        for i, value in enumerate(weights):
            plt.text(i, value + (0.1 * (-1 if value <= 0 else 1)),
                     str(round(value, 3)), ha='center', fontsize=6)
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Feature Attribution Weights - SM Order')
        plt.show()
