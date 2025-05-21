import torch
import torch.nn.functional as F
from deeparguing.base_scores.compute_base_scores import ComputeBaseScores
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor
from typing import List


class LearnedBaseScore(ComputeBaseScores):

    def __init__(self,  feature_extractors: List[FeatureExtractor], activation=lambda x: x, temperature = 1., batch_size = None):
        super(LearnedBaseScore, self).__init__()

        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.activation = activation
        self.temperature = temperature
        self.batch_size = batch_size


    def forward(self, features: torch.Tensor) -> torch.Tensor:

        for feature_extractor in self.feature_extractors:
            if self.batch_size == None:
                features = feature_extractor(features)
            else:
                features = self.split_apply_fe(features, feature_extractor, self.batch_size)
        return self.activation(features/self.temperature) 


    def plot_parameters(self):
        for feature_extractor in self.feature_extractors:
            feature_extractor.plot_parameters()


    def split_apply_fe(self, cases, feature_extractor, batch_size):
        split_cases = torch.split(cases, batch_size)
        result = [feature_extractor(cases_i) for cases_i in split_cases]
        result = torch.cat(result, dim=0)
        return result