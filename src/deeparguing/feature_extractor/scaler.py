import torch
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor

class Scaler(FeatureExtractor):


    def __init__(self, no_features, mean=0, std=1, weight=None):
        super(Scaler, self).__init__(no_features)
        if weight == None:
            self.W = torch.nn.Parameter(torch.Tensor(1))
            torch.nn.init.normal_(self.W, mean=mean, std=std)
        else:
            self.W = torch.nn.Parameter(torch.tensor(weight))

    def forward(self, case: torch.tensor) -> torch.tensor:
        return torch.mul(case, self.W)


    def get_output_features(self) -> int:
        return self.no_features 

    def plot_parameters(self):
        print("SCALE FACTOR", self.W)
