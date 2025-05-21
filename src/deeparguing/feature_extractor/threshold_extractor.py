import torch
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor



class ThresholdFeatureExtractor(FeatureExtractor):

    def __init__(self, no_features, signed=True):
        super(ThresholdFeatureExtractor, self).__init__(no_features)
        self.signed = signed
        self.thresholds = torch.nn.Parameter(torch.Tensor(no_features))
        torch.nn.init.normal_(self.thresholds)

    def forward(self, case: torch.Tensor):
        thresholded = torch.relu(torch.abs(case) - torch.abs(self.thresholds))
        if self.signed:
            result = torch.mul(thresholded, case.sign())
        result = thresholded
        return torch.sum(result, dim=-1)




    def get_output_features(self) -> int:
        return self.no_features

    def plot_parameters(self):
        thresholds = self.thresholds.detach().cpu().numpy()
        plt.figure(figsize=(20, 5))
        plt.bar(range(len(thresholds)), thresholds)
        for i, value in enumerate(thresholds):
            plt.text(i, value + (0.1 * (-1 if value <= 0 else 1)),
                     str(round(value, 3)), ha='center', fontsize=6)
        plt.xlabel('Features')
        plt.ylabel('Threshold Values')
        plt.title('Feature Thresholds')
        plt.show()

