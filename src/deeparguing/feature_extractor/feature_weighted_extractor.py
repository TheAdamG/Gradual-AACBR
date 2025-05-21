import torch
import torch.nn.init as init
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor

class FeatureWeightedExtractor(FeatureExtractor):


    def __init__(self, no_features, inital_weights=None, initialisation_method=lambda x: init.xavier_uniform_(x.unsqueeze(1))):
        super(FeatureWeightedExtractor, self).__init__(no_features)
        if inital_weights is None:
            self.W = torch.nn.Parameter(torch.Tensor(no_features))
            initialisation_method(self.W)
        else:
            self.W = torch.nn.Parameter(inital_weights) 
        

    def forward(self, case: torch.Tensor) -> torch.Tensor:
        W = self.W.squeeze()
        return torch.matmul(case, W)


    def get_output_features(self) -> int:
        return self.no_features

    def plot_parameters(self):
        weights = self.W.squeeze().detach().cpu().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(weights)), weights)
        for i, value in enumerate(weights):
            plt.text(i, value + (0.1 * (-1 if value <= 0 else 1)),
                     str(round(value, 3)), ha='center', fontsize=6)
        plt.xlabel('Features')
        plt.ylabel('Weights')
        plt.title('Feature Attribution Weights')
        plt.show()
